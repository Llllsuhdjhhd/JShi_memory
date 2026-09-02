from __future__ import annotations

from typing import Optional

# 仓储层：Event/Role/Metabolism 的 CRUD 与 ORM ↔ Pydantic 模型转换。

from ..models.event import EmotionalModel, Event, EventRoleEntry, EventStatus
from ..models.metabolism import Shadow, UnclosedEvent
from ..models.object_entry import ObjectMemoryEntry
from ..models.role import Role, WhitePaintingEntry
from datetime import datetime

from ..port import RecallTrace
from .database import (
    Database,
    EventRecord,
    ObjectMemoryEntryRecord,
    RecallTraceRecord,
    RoleRecord,
    ShadowRecord,
    StoredMarkRecord,
    UnclosedEventRecord,
    WhitePaintingRecord,
)


# =====================================================================
# Event CRUD
# =====================================================================

class EventRepository:
    def __init__(self, db: Database):
        self._db = db

    def save(self, event: Event) -> None:
        with self._db.session() as s:
            record = EventRecord(
                event_id=event.event_id,
                subject_id=getattr(event, "subject_id", "") or "",
                create_time=event.create_time,
                occurred_at=event.occurred_at,
                source_ids=list(event.source_ids or []),
                content_raw=event.content_raw,
                summaries=event.summaries,
                summary_lengths=event.summary_lengths,
                actual_max_level=event.actual_max_level,
                role_list=[r.model_dump(mode="json") for r in event.role_list],
                is_abstract=event.is_abstract,
                is_abstracted=event.is_abstracted,
                status=event.status.value,
                decoration=event.decoration,
                insight=event.insight,
                event_length=event.event_length,
                abstraction_level=event.abstraction_level,
                source_events=event.source_events,
                is_tombstoned=event.is_tombstoned,
                activation_energy=event.activation_energy,
                compression_ratio=event.compression_ratio,
                abstract_coverage=float(getattr(event, "abstract_coverage", 0.0) or 0.0),
                split_successor_event_ids=list(event.split_successor_event_ids or []),
                split_prefix_event_ids=list(event.split_prefix_event_ids or []),
                ptsd_immune=bool(getattr(event, "ptsd_immune", False)),
                origin=getattr(event, "origin", None) or "external",
                location=getattr(event, "location", None),
                emotion=event.emotion.model_dump(mode="json") if event.emotion is not None else None,
                forgetting_factor=float(getattr(event, "forgetting_factor", 1.0) or 1.0),
                recall_metadata={
                    "keywords": list(getattr(event, "keywords", None) or []),
                    "location": getattr(event, "location", None),
                    "interlocutor": getattr(event, "interlocutor", None) or None,
                },
            )
            s.merge(record)
            s.commit()

    def append_split_successor(self, event_id: str, successor_event_id: str) -> None:
        """Append *successor_event_id* to ``events.split_successor_event_ids`` (dedup).

        在 tail UC 闭环封存为事件后，反向把该 event_id 记到其**每一个**前缀事件上，
        以便审计 / 观测侧能从 "前缀"出发查到 "后继"。去重 + 保持顺序；目标记录
        不存在时静默返回。
        """
        with self._db.session() as s:
            r = s.get(EventRecord, event_id)
            if r is None:
                return
            current = list(r.split_successor_event_ids or [])
            if successor_event_id in current:
                return
            current.append(successor_event_id)
            r.split_successor_event_ids = current
            s.commit()

    def get(self, event_id: str) -> Optional[Event]:
        with self._db.session() as s:
            r = s.get(EventRecord, event_id)
            return self._to_model(r) if r else None

    def list_all(
        self,
        *,
        is_abstract: bool | None = None,
        is_abstracted: bool | None = None,
        status: EventStatus | None = None,
        exclude_tombstoned: bool = True,
    ) -> list[Event]:
        with self._db.session() as s:
            q = s.query(EventRecord)
            if is_abstract is not None:
                q = q.filter(EventRecord.is_abstract == is_abstract)
            if is_abstracted is not None:
                q = q.filter(EventRecord.is_abstracted == is_abstracted)
            if status is not None:
                q = q.filter(EventRecord.status == status.value)
            if exclude_tombstoned:
                q = q.filter(EventRecord.is_tombstoned == False)  # noqa: E712
            return [self._to_model(r) for r in q.order_by(EventRecord.create_time).all()]

    def count(
        self,
        *,
        is_abstract: bool | None = None,
        exclude_tombstoned: bool = True,
    ) -> int:
        """Count events with optional filters (used by 70/30 capacity logic and panels)."""
        with self._db.session() as s:
            q = s.query(EventRecord)
            if is_abstract is not None:
                q = q.filter(EventRecord.is_abstract == is_abstract)
            if exclude_tombstoned:
                q = q.filter(EventRecord.is_tombstoned == False)  # noqa: E712
            return q.count()

    def find_time_boundary(self, capacity: int, global_ratio: float) -> "datetime | None":
        """Return the create_time that splits the active basic-event population into
        最近 *global_ratio* (e.g. 70%) 与较早的 30% 两段。

        70/30 分层检索（白皮书 §4.4 Lazy Index 容量分层）需要这条边界：
            - 全库非墓碑、非抽象事件按 ``create_time`` 升序；
            - 取末尾 ``ratio = global_ratio`` 段的起点 create_time 即为边界；
            - 库容量低于 ``capacity`` 时返回 ``None``，调用方应回退到全局检索。

        计算方式与 ``recall_max_capacity`` 解耦——边界总是按 *当前活跃总数* 与
        *global_ratio* 计算，不被 ``capacity`` 截断；后者只决定"是否启用分层"。
        """
        if capacity <= 0:
            return None
        with self._db.session() as s:
            q = (
                s.query(EventRecord.create_time)
                .filter(EventRecord.is_abstract == False)  # noqa: E712
                .filter(EventRecord.is_tombstoned == False)  # noqa: E712
                .order_by(EventRecord.create_time.asc())
            )
            total = q.count()
            if total <= capacity:
                return None
            # 最近 global_ratio 段从索引 split_idx 开始
            split_idx = max(0, int(total * (1.0 - global_ratio)))
            row = q.offset(split_idx).limit(1).first()
            return row[0] if row else None

    def resolve_basic_event_ids(self, event_id: str) -> list[str]:
        """Return the flat list of basic-event IDs reachable from *event_id*.

        抽象事件可作为更高阶抽象的 ``source_events`` 成员再次参与合成（白皮书 §3.2），
        因此 ``source_events`` 可能嵌套抽象事件。本方法按广度优先展开抽象链，
        收集所有叶子层的基本事件 ID（去重、保持首次发现顺序）：
            - 起点若是基本事件：返回 ``[event_id]``；
            - 起点是抽象事件：递归展开其 ``source_events``，逐层下钻直到全部叶子为基本事件；
            - 遇到已登记为墓碑的事件会被跳过（不进入结果，但仍继续展开其他分支）；
            - 缺失 / 循环引用自动通过 ``seen`` 集合短路，不会无限递归。

        用途：从一条抽象事件出发做"证据溯源"——回到具体的基本事件层，用于高保真复盘、
        审计、或在 UI 中提供"展开到原文"入口。
        """
        seen: set[str] = set()
        result: list[str] = []
        stack: list[str] = [event_id]
        with self._db.session() as s:
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                r = s.get(EventRecord, cur)
                if r is None or r.is_tombstoned:
                    continue
                if r.is_abstract:
                    # 抽象事件：推入所有 source_events，继续下钻。
                    for sid in list(r.source_events or []):
                        if sid not in seen:
                            stack.append(sid)
                else:
                    # 基本事件：加入结果。
                    result.append(r.event_id)
        return result

    def apply_abstract_coverage_increment(
        self,
        event_ids: set[str],
        delta: float,
        cap: float,
    ) -> None:
        """Add *delta* to ``abstract_coverage`` for each id in *event_ids*, hard-capped at *cap*.

        新抽象 A 合成后对覆盖集内所有基本事件及被取代的中间抽象节点一次性累加；
        多次抽象路径上会反复递增直至封顶。
        """
        if not event_ids or delta <= 0:
            return
        with self._db.session() as s:
            for eid in event_ids:
                r = s.get(EventRecord, eid)
                if r is None or getattr(r, "is_tombstoned", False):
                    continue
                cur = float(getattr(r, "abstract_coverage", 0.0) or 0.0)
                r.abstract_coverage = min(float(cap), cur + float(delta))
            s.commit()

    def update_status(
        self,
        event_id: str,
        *,
        is_abstracted: bool | None = None,
        status: EventStatus | None = None,
        is_tombstoned: bool | None = None,
    ) -> None:
        with self._db.session() as s:
            r = s.get(EventRecord, event_id)
            if not r:
                return
            if is_abstracted is not None:
                r.is_abstracted = is_abstracted
            if status is not None:
                r.status = status.value
            if is_tombstoned is not None:
                r.is_tombstoned = is_tombstoned
            s.commit()

    def update_forgetting_factor(self, event_id: str, factor: float) -> None:
        """记忆单元级遗忘因子更新（回忆命中的记忆恢复，design/1010）。"""
        with self._db.session() as s:
            r = s.get(EventRecord, event_id)
            if r is None:
                return
            r.forgetting_factor = float(factor)
            s.commit()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_model(r: EventRecord) -> Event:
        role_list = [EventRoleEntry(**rd) for rd in (r.role_list or [])]
        return Event(
            event_id=r.event_id,
            subject_id=getattr(r, "subject_id", "") or "",
            create_time=r.create_time,
            occurred_at=getattr(r, "occurred_at", None),
            source_ids=list(getattr(r, "source_ids", None) or []),
            content_raw=r.content_raw,
            summaries=r.summaries or {},
            summary_lengths=r.summary_lengths or {},
            actual_max_level=r.actual_max_level or 0,
            role_list=role_list,
            is_abstract=r.is_abstract,
            is_abstracted=r.is_abstracted,
            status=EventStatus(r.status),
            decoration=r.decoration,
            insight=r.insight,
            event_length=r.event_length or 0,
            abstraction_level=r.abstraction_level,
            source_events=r.source_events,
            is_tombstoned=bool(r.is_tombstoned),
            activation_energy=float(r.activation_energy or 0.0),
            compression_ratio=float(r.compression_ratio or 0.0),
            abstract_coverage=float(getattr(r, "abstract_coverage", 0.0) or 0.0),
            split_successor_event_ids=list(getattr(r, "split_successor_event_ids", None) or []),
            split_prefix_event_ids=list(getattr(r, "split_prefix_event_ids", None) or []),
            ptsd_immune=bool(getattr(r, "ptsd_immune", False)),
            origin="external" if (getattr(r, "origin", None) or "") in ("", "normal") else r.origin,
            keywords=list((getattr(r, "recall_metadata", None) or {}).get("keywords") or []),
            location=getattr(r, "location", None) or (getattr(r, "recall_metadata", None) or {}).get("location"),
            interlocutor=(getattr(r, "recall_metadata", None) or {}).get("interlocutor") or None,
            emotion=EmotionalModel(**r.emotion) if (getattr(r, "emotion", None) or None) else None,
            forgetting_factor=float(getattr(r, "forgetting_factor", 1.0) or 1.0),
        )


# =====================================================================
# Role CRUD
# =====================================================================

class RoleRepository:
    def __init__(self, db: Database):
        self._db = db

    def save(self, role: Role) -> None:
        with self._db.session() as s:
            record = RoleRecord(
                role_id=role.role_id,
                name=role.name,
                entity_type=role.entity_type,
                aliases=role.aliases,
                created_at=role.created_at,
                is_suspicious=role.is_suspicious,
            )
            s.merge(record)
            s.commit()

    def get(self, role_id: str) -> Optional[Role]:
        with self._db.session() as s:
            record = s.get(RoleRecord, role_id)
            if not record:
                return None
            entries = (
                s.query(WhitePaintingRecord)
                .filter(WhitePaintingRecord.role_id == role_id)
                .order_by(WhitePaintingRecord.create_time)
                .all()
            )
            return self._build_role(record, entries)

    def list_all(self) -> list[Role]:
        with self._db.session() as s:
            records = s.query(RoleRecord).order_by(RoleRecord.created_at).all()
            result: list[Role] = []
            for rec in records:
                entries = (
                    s.query(WhitePaintingRecord)
                    .filter(WhitePaintingRecord.role_id == rec.role_id)
                    .order_by(WhitePaintingRecord.create_time)
                    .all()
                )
                result.append(self._build_role(rec, entries))
            return result

    def find_by_name(self, name: str) -> Optional[Role]:
        """Find a role by canonical name or alias.

        Two-stage lookup:
            1. SQL equality on ``name`` (indexed enough for typical scales).
            2. Fallback alias scan with **lazy iteration** + early stop —
               不再对每条候选 role 都触发一次 ``self.get`` 的额外查询，避免 O(N²)
               的级联 round-trip（白描数据量大时影响显著）。
        """
        with self._db.session() as s:
            record = s.query(RoleRecord).filter(RoleRecord.name == name).first()
            if record:
                role_id = record.role_id
            else:
                role_id = None
                for r in s.query(RoleRecord).yield_per(200):
                    if name in (r.aliases or []):
                        role_id = r.role_id
                        break
            if role_id is None:
                return None
        return self.get(role_id)

    def add_white_painting_entry(self, role_id: str, entry: WhitePaintingEntry) -> None:
        with self._db.session() as s:
            importance_val = entry.importance.value if hasattr(entry.importance, "value") else str(entry.importance)
            record = WhitePaintingRecord(
                role_id=role_id,
                subject_id=entry.subject_id,
                event_id=entry.event_id,
                role_summary=entry.role_summary,
                l1_mention=entry.l1_mention,
                l2_interaction=entry.l2_interaction,
                l3_decision=entry.l3_decision,
                emotional_model=entry.emotional_model.model_dump(mode="json"),
                importance=importance_val,
                create_time=entry.create_time,
                memory_weight=entry.memory_weight,
                forgetting_factor=entry.forgetting_factor,
                base_forgetting_factor=entry.base_forgetting_factor,
                last_accessed_time=entry.last_accessed_time,
                is_suspicious=entry.is_suspicious,
            )
            s.add(record)
            s.commit()

    def get_white_painting(
        self,
        role_id: str,
        *,
        limit: int | None = None,
        min_memory_weight: float | None = None,
    ) -> list[WhitePaintingEntry]:
        with self._db.session() as s:
            q = (
                s.query(WhitePaintingRecord)
                .filter(WhitePaintingRecord.role_id == role_id)
                .order_by(WhitePaintingRecord.create_time)
            )
            if min_memory_weight is not None:
                q = q.filter(WhitePaintingRecord.memory_weight >= min_memory_weight)
            if limit:
                q = q.limit(limit)
            return [self._to_wp_entry(e) for e in q.all()]

    def get_white_painting_by_event(self, role_id: str, event_id: str) -> WhitePaintingEntry | None:
        with self._db.session() as s:
            record = (
                s.query(WhitePaintingRecord)
                .filter(
                    WhitePaintingRecord.role_id == role_id,
                    WhitePaintingRecord.event_id == event_id
                )
                .first()
            )
            if record is None:
                return None
            return self._to_wp_entry(record)

    def update_white_painting_access(self, role_id: str, event_id: str, new_forgetting_factor: float) -> None:
        from datetime import datetime
        with self._db.session() as s:
            record = (
                s.query(WhitePaintingRecord)
                .filter(
                    WhitePaintingRecord.role_id == role_id,
                    WhitePaintingRecord.event_id == event_id
                )
                .first()
            )
            if record:
                record.forgetting_factor = new_forgetting_factor
                record.last_accessed_time = datetime.now()
                s.commit()

    def get_wp_total_length(self, role_id: str) -> int:
        """Return the total character length of all white-painting entries for a role.

        用于角色容量判定（白皮书 2.3）：当总长超过 ``wp_role_capacity`` 时触发软遗忘。
        """
        with self._db.session() as s:
            from sqlalchemy import func
            result = s.query(func.sum(func.length(WhitePaintingRecord.role_summary))).filter(
                WhitePaintingRecord.role_id == role_id
            ).scalar()
            return int(result or 0)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_role(
        rec: RoleRecord,
        entries: list[WhitePaintingRecord],
    ) -> Role:
        return Role(
            role_id=rec.role_id,
            name=rec.name,
            entity_type=rec.entity_type,
            aliases=rec.aliases or [],
            created_at=rec.created_at,
            white_painting=[RoleRepository._to_wp_entry(e) for e in entries],
            is_suspicious=bool(rec.is_suspicious),
        )

    @staticmethod
    def _to_wp_entry(e: WhitePaintingRecord) -> WhitePaintingEntry:
        em = EmotionalModel(**(e.emotional_model or {}))
        return WhitePaintingEntry(
            subject_id=getattr(e, "subject_id", "") or "",
            event_id=e.event_id,
            role_summary=e.role_summary,
            l1_mention=getattr(e, "l1_mention", None),
            l2_interaction=getattr(e, "l2_interaction", None),
            l3_decision=getattr(e, "l3_decision", None),
            emotional_model=em,
            importance=e.importance,
            create_time=e.create_time,
            memory_weight=float(e.memory_weight or 0.0),
            forgetting_factor=float(getattr(e, "forgetting_factor", 1.0) or 1.0),
            base_forgetting_factor=float(getattr(e, "base_forgetting_factor", 1.0) or 1.0),
            last_accessed_time=getattr(e, "last_accessed_time", e.create_time) or e.create_time,
            is_suspicious=bool(e.is_suspicious),
        )


# =====================================================================
# Metabolism state (shadow + unclosed events)
# =====================================================================

class MetabolismRepository:
    def __init__(self, db: Database):
        self._db = db

    def get_shadow(self) -> Shadow:
        with self._db.session() as s:
            record = s.query(ShadowRecord).first()
            if not record:
                return Shadow()
            return Shadow(
                content=record.content,
                updated_at=record.updated_at,
                subject_id=getattr(record, "subject_id", "") or "",
            )

    def update_shadow(self, shadow: Shadow) -> None:
        with self._db.session() as s:
            record = s.query(ShadowRecord).first()
            if record:
                record.content = shadow.content
                record.updated_at = shadow.updated_at
                record.subject_id = shadow.subject_id
            else:
                s.add(ShadowRecord(
                    content=shadow.content,
                    updated_at=shadow.updated_at,
                    subject_id=shadow.subject_id,
                ))
            s.commit()

    def save_unclosed_event(self, event: UnclosedEvent) -> None:
        with self._db.session() as s:
            record = UnclosedEventRecord(
                id=event.id,
                subject_id=event.subject_id,
                content_fragments=event.content_fragments,
                identified_roles=event.identified_roles,
                logical_gaps=event.logical_gaps,
                created_at=event.created_at,
                updated_at=event.updated_at,
                last_hit_time=event.last_hit_time,
                split_prefix_event_ids=list(event.split_prefix_event_ids or []),
                oversized=bool(event.oversized),
            )
            s.merge(record)
            s.commit()

    def get_unclosed_events(self) -> list[UnclosedEvent]:
        with self._db.session() as s:
            return [self._to_model(r) for r in s.query(UnclosedEventRecord).all()]

    def get_unclosed_event(self, event_id: str) -> Optional[UnclosedEvent]:
        with self._db.session() as s:
            r = s.get(UnclosedEventRecord, event_id)
            return self._to_model(r) if r else None

    def delete_unclosed_event(self, event_id: str) -> None:
        with self._db.session() as s:
            s.query(UnclosedEventRecord).filter(UnclosedEventRecord.id == event_id).delete()
            s.commit()

    @staticmethod
    def _to_model(r: UnclosedEventRecord) -> UnclosedEvent:
        return UnclosedEvent(
            id=r.id,
            subject_id=getattr(r, "subject_id", "") or "",
            content_fragments=r.content_fragments or [],
            identified_roles=r.identified_roles or [],
            logical_gaps=r.logical_gaps,
            created_at=r.created_at,
            updated_at=r.updated_at,
            last_hit_time=r.last_hit_time,
            split_prefix_event_ids=list(getattr(r, "split_prefix_event_ids", None) or []),
            oversized=bool(getattr(r, "oversized", False) or False),
        )


# =====================================================================
# Object memory timeline（design/410/810；无情感，只存摘要轨迹）
# =====================================================================

class ObjectTimelineRepository:
    """对象记忆时间线条目：object_id + name + 一句摘要 + 时间。"""

    def __init__(self, db: Database):
        self._db = db

    def append(self, entry: ObjectMemoryEntry) -> None:
        with self._db.session() as s:
            s.add(ObjectMemoryEntryRecord(
                subject_id=entry.subject_id,
                object_id=entry.object_id,
                name=entry.name,
                event_id=entry.event_id,
                summary=entry.summary,
                create_time=entry.create_time,
            ))
            s.commit()

    def list_by_object(
        self, subject_id: str, object_id: str, *, limit: int | None = None
    ) -> list[ObjectMemoryEntry]:
        with self._db.session() as s:
            q = s.query(ObjectMemoryEntryRecord).filter(
                ObjectMemoryEntryRecord.subject_id == subject_id,
                ObjectMemoryEntryRecord.object_id == object_id,
            ).order_by(ObjectMemoryEntryRecord.create_time)
            if limit is not None:
                q = q.limit(limit)
            return [self._to_model(r) for r in q.all()]

    def list_by_event(self, event_id: str) -> list[ObjectMemoryEntry]:
        with self._db.session() as s:
            rows = (
                s.query(ObjectMemoryEntryRecord)
                .filter(ObjectMemoryEntryRecord.event_id == event_id)
                .order_by(ObjectMemoryEntryRecord.create_time)
                .all()
            )
            return [self._to_model(r) for r in rows]

    @staticmethod
    def _to_model(r: ObjectMemoryEntryRecord) -> ObjectMemoryEntry:
        return ObjectMemoryEntry(
            subject_id=getattr(r, "subject_id", "") or "",
            object_id=r.object_id,
            name=r.name,
            event_id=r.event_id,
            summary=r.summary,
            create_time=r.create_time,
        )


# =====================================================================
# stored_marks 台账（design/210/810；30 侧推进游标用，纯台账）
# =====================================================================

class StoredMarksRepository:
    """segment_id（经历段 id）→ 本轮封存事件 id 列表。"""

    def __init__(self, db: Database):
        self._db = db

    def save(self, subject_id: str, input_id: str, event_ids: list[str]) -> None:
        with self._db.session() as s:
            s.merge(StoredMarkRecord(
                subject_id=subject_id,
                input_id=input_id,
                event_ids=list(event_ids),
                created_at=datetime.now(),
            ))
            s.commit()

    def get(self, input_id: str) -> list[str] | None:
        with self._db.session() as s:
            r = s.get(StoredMarkRecord, input_id)
            return list(r.event_ids or []) if r else None

    def list_all(self, subject_id: str | None = None) -> list[dict]:
        with self._db.session() as s:
            q = s.query(StoredMarkRecord)
            if subject_id:
                q = q.filter(StoredMarkRecord.subject_id == subject_id)
            return [
                {
                    "subject_id": getattr(r, "subject_id", "") or "",
                    "input_id": r.input_id,
                    "event_ids": list(r.event_ids or []),
                    "created_at": r.created_at,
                }
                for r in q.order_by(StoredMarkRecord.created_at).all()
            ]


# =====================================================================
# recall_traces（design/1010 可观测：本次回忆输入 + 命中条目）
# =====================================================================

class RecallTraceRepository:
    """recall_traces 表读写：一次 recall 调用一行（输入 + 命中条目）。

    用于回溯召回质量与说话人归属（诊断张冠李戴）；只作观测，不影响召回结果。
    """

    def __init__(self, db: Database):
        self._db = db

    def save(self, trace: RecallTrace) -> None:
        with self._db.session() as s:
            record = RecallTraceRecord(
                recall_id=trace.recall_id,
                subject_id=trace.subject_id,
                query=trace.query,
                object_id=trace.object_id,
                level=trace.level,
                limit=trace.limit,
                anchor_event_ids=list(trace.anchor_event_ids or []),
                created_at=trace.created_at,
                items=[it.model_dump(mode="json") for it in trace.items],
                n_items=len(trace.items),
            )
            s.merge(record)
            s.commit()

    def get(self, recall_id: str) -> dict | None:
        with self._db.session() as s:
            r = s.get(RecallTraceRecord, recall_id)
            return self._to_dict(r) if r else None

    def list_all(
        self,
        subject_id: str | None = None,
        *,
        limit: int | None = None,
        ascending: bool = False,
    ) -> list[dict]:
        with self._db.session() as s:
            q = s.query(RecallTraceRecord)
            if subject_id:
                q = q.filter(RecallTraceRecord.subject_id == subject_id)
            order = (
                RecallTraceRecord.created_at.asc() if ascending
                else RecallTraceRecord.created_at.desc()
            )
            q = q.order_by(order)
            if limit:
                q = q.limit(int(limit))
            return [self._to_dict(r) for r in q.all()]

    @staticmethod
    def _to_dict(r: RecallTraceRecord) -> dict:
        return {
            "recall_id": r.recall_id,
            "subject_id": getattr(r, "subject_id", "") or "",
            "query": r.query,
            "object_id": r.object_id,
            "level": r.level,
            "limit": r.limit,
            "anchor_event_ids": list(r.anchor_event_ids or []),
            "created_at": r.created_at,
            "items": list(r.items or []),
            "n_items": int(r.n_items or 0),
        }
