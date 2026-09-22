"""REMSPipeline：匠石记忆后端（REMS3 fork）新编排。

- **记忆写入**：``ingest_batch``（MemoryBackendPort）——逐 input 代谢封存，写对象时间线 /
  stored_marks / 向量索引（design/210、610、810、1010）；
- **回忆读取**：``recall``（MemoryBackendPort）——RecallPipeline 多路召回（design/1010）；
- **信念修正**：``tombstone``——保留墓碑标记，只降可见性、不删原文（design/10）。
"""

from __future__ import annotations

import logging
import uuid

from .config import REMSConfig
from .llm.provider import LLMProvider
from .models.event import Event
from .port import (
    BackendIngestResult,
    MemoryBatch,
    MemoryExperience,
    RecalledFragment,
    RecallTrace,
    RecallTraceItem,
    SubSegment,
)
from .recall import (
    BgeReranker,
    HashEmbedding,
    NullReranker,
    QdrantRecallVectorStore,
    RecallPipeline,
    SentenceTransformerEmbedding,
)
from .recall.intent import RuleIntentClassifier
from .utils.text import segment_sentences
from .services.belief_revision_service import BeliefRevisionService
from .services.emotion_service import EMAEvolver
from .services.event_service import EventService
from .services.metabolism_service import MetabolismService
from .services.portrait_service import PortraitService
from .services.recall_budget import RecallBudgetManager
from .services.role_service import RoleService
from .skills.boundary_detection import BoundaryDetectionSkill
from .skills.event_enrichment import EventEnrichmentSkill
from .skills.portrait_compression import PortraitCompressionSkill
from .skills.role_extraction import RoleExtractionSkill
from .storage.database import Database
from .storage.repository import (
    EventRepository,
    MetabolismRepository,
    ObjectTimelineRepository,
    PortraitRepository,
    RecallTraceRepository,
    RoleRepository,
    StoredMarksRepository,
)
from .strategies.event_weight import EventWeightDeriver

logger = logging.getLogger(__name__)


class REMSPipeline:
    """记忆写入 + 回忆读取双端口编排。"""

    def __init__(
        self,
        config: REMSConfig,
        llm: LLMProvider,
        db: Database,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        meta_repo: MetabolismRepository,
        event_service: EventService,
        role_service: RoleService,
        metabolism_service: MetabolismService,
        belief_revision_service: BeliefRevisionService,
        *,
        recall_pipeline: RecallPipeline | None = None,
        object_timeline_repo: ObjectTimelineRepository | None = None,
        stored_marks_repo: StoredMarksRepository | None = None,
        recall_trace_repo: RecallTraceRepository | None = None,
        portrait_service: PortraitService | None = None,
        recall_budget: RecallBudgetManager | None = None,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.event_repo = event_repo
        self.role_repo = role_repo
        self.meta_repo = meta_repo
        self.event_service = event_service
        self.role_service = role_service
        self.metabolism_service = metabolism_service
        self.belief_revision_service = belief_revision_service
        self.recall_pipeline = recall_pipeline
        self.object_timeline_repo = object_timeline_repo
        self.stored_marks_repo = stored_marks_repo
        self.recall_trace_repo = recall_trace_repo
        self.portrait_service = portrait_service
        self.recall_budget = recall_budget

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: REMSConfig | None = None) -> "REMSPipeline":
        """工厂：装配存储、技能、领域服务与回忆 provider（插拔式，design/1010 §9.1）。"""
        if config is None:
            config = REMSConfig()

        llm = LLMProvider(config)
        db = Database(config.storage.database_url)
        db.create_tables()

        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        meta_repo = MetabolismRepository(db)
        object_timeline_repo = ObjectTimelineRepository(db)
        stored_marks_repo = StoredMarksRepository(db)
        recall_trace_repo = RecallTraceRepository(db)

        portrait_service: PortraitService | None = None
        if config.portrait_enabled:
            portrait_repo = PortraitRepository(db)
            portrait_skill = PortraitCompressionSkill(llm, config)
            portrait_service = PortraitService(config, portrait_repo, portrait_skill)

        recall_budget = RecallBudgetManager(config) if config.recall_budget_enabled else None

        # ---- 回忆 provider（插拔式） ----
        if config.embedding.provider == "hash":
            embedding = HashEmbedding()
        else:
            embedding = SentenceTransformerEmbedding(config.embedding.model_name)
        vector_store = QdrantRecallVectorStore(
            config.storage.qdrant_collection,
            url=config.storage.qdrant_url,
            path=config.storage.qdrant_path,
        )
        # reranker 默认 Null（纯 RRF）；启用时换 BgeReranker（见配置项，后续接入）。
        reranker: BgeReranker | NullReranker = NullReranker()
        intent_classifier = RuleIntentClassifier(
            entity_lexicon=config.recall_intent_entity_lexicon,
            emotion_lexicon=config.recall_intent_emotion_lexicon,
            fact_lexicon=config.recall_intent_fact_lexicon,
        )
        recall_pipeline = RecallPipeline(
            embedding,
            vector_store,
            event_repo,
            object_timeline_repo,
            reranker=reranker,
            intent_classifier=intent_classifier,
            rrf_k=config.recall_rrf_k,
            reinforce_multiplier=config.recall_reinforce_multiplier,
            reinforce_cap=config.recall_forgetting_factor_cap,
            recency_enabled=config.recall_recency_enabled,
            recency_window_events=config.recall_recency_window_events,
            recency_top_k=config.recall_recency_top_k,
            factor_alpha=config.recall_factor_alpha,
            mood_beta=config.recall_mood_beta,
            object_affinity_enabled=config.recall_object_affinity_enabled,
            object_affinity_boost=config.recall_object_affinity_boost,
            object_affinity_penalty=config.recall_object_affinity_penalty,
            event_fatigue=config.event_fatigue,
            meta_repo=meta_repo,
            include_unclosed=config.recall_include_unclosed,
            unclosed_same_object_score=config.recall_unclosed_same_object_score,
        )

        # ---- 技能与领域服务 ----
        role_skill = RoleExtractionSkill(llm, config)
        boundary_skill = BoundaryDetectionSkill(llm, config)
        enrichment_skill = EventEnrichmentSkill(llm, config, role_fallback=role_skill)

        event_weight = EventWeightDeriver(config, event_repo, role_repo)
        emotion_evolver = EMAEvolver(config, role_repo)
        role_service = RoleService(config, role_repo, role_skill, llm=llm)

        event_service = EventService(
            config,
            llm,
            event_repo,
            None,  # 旧向量库不再使用（design/810：向量占位 → 回忆由 recall_pipeline 索引）
            enrichment_skill,
            role_service=role_service,
            emotion_evolver=emotion_evolver,
            event_weight=event_weight,
            object_timeline_repo=object_timeline_repo,
        )
        metabolism_service = MetabolismService.with_default_boundary_repair(
            config,
            meta_repo,
            boundary_skill,
            event_service,
            event_repo=event_repo,
            llm=llm,
        )
        belief_revision_service = BeliefRevisionService(config, event_repo, role_repo)

        return cls(
            config=config,
            llm=llm,
            db=db,
            event_repo=event_repo,
            role_repo=role_repo,
            meta_repo=meta_repo,
            event_service=event_service,
            role_service=role_service,
            metabolism_service=metabolism_service,
            belief_revision_service=belief_revision_service,
            recall_pipeline=recall_pipeline,
            object_timeline_repo=object_timeline_repo,
            stored_marks_repo=stored_marks_repo,
            recall_trace_repo=recall_trace_repo,
            portrait_service=portrait_service,
            recall_budget=recall_budget,
        )

    # ------------------------------------------------------------------
    # 记忆写入端口（design/210/610）
    # ------------------------------------------------------------------

    def ingest_batch(self, batch: MemoryBatch) -> BackendIngestResult:
        """记忆写入：整批合并为一段输入走一次代谢（与 rems3 整批语义对齐），
        再按事件内容归属回各 segment 写 stored_marks / 对象时间线 / 向量索引。
        空文本段按契约报错并跳过；单批失败进 ``errors``。
        """
        result = BackendIngestResult(subject_id=batch.subject_id)

        valid: list[MemoryExperience] = []
        for inp in batch.experiences:
            if not (inp.text or "").strip():
                result.errors.append(f"segment {inp.segment_id or '?'}: empty text")
                continue
            valid.append(inp)

        if valid:
            # 整批合并：轮次之间空行分隔，保留“匠石:”/“deepseek:”前缀
            combined_text = "\n\n".join(e.text for e in valid)
            merged_objects: dict[str, str] = {}
            merged_sources: list[str] = []
            sub_segments: list[SubSegment] = []
            for e in valid:
                merged_objects.update(e.objects)
                merged_sources.extend(e.source_ids)
                sub_segments.append(
                    SubSegment(
                        segment_id=e.segment_id or f"seg-{len(sub_segments)+1:03d}",
                        text=e.text,
                        objects=dict(e.objects),
                        interlocutor=getattr(e, "interlocutor", None) or None,
                    )
                )
            memory = MemoryExperience(
                subject_id=batch.subject_id,
                text=combined_text,
                objects=merged_objects,
                source_ids=tuple(merged_sources),
                segment_id="+".join(e.segment_id for e in valid),
                origin=valid[0].origin,
                sub_segments=sub_segments,
                interlocutor=next(
                    (getattr(e, "interlocutor", None) for e in valid if getattr(e, "interlocutor", None)),
                    None,
                ),
            )
            try:
                sealed = self.metabolism_service.process_input(
                    combined_text, memory=memory,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ingest_batch failed: %s", exc)
                result.errors.append(f"batch ingest: {exc}")
                sealed = []

            for ev in sealed:
                result.sealed_event_ids.append(ev.event_id)
                for re_ in ev.role_list:
                    if (
                        not re_.is_subject
                        and re_.role_id
                        and re_.role_id not in result.role_ids
                    ):
                        result.role_ids.append(re_.role_id)
                if self.recall_pipeline is not None:
                    try:
                        self.recall_pipeline.index_event(ev)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("index_event %s failed: %s", ev.event_id, exc)
                        result.errors.append(f"index {ev.event_id}: {exc}")

            # stored_marks：按事件内容归属回各 segment（一个事件可命中多个 segment；
            # 段被部分封存时按句子级重叠判定，避免漏归属）
            def _norm(s: str) -> str:
                return "".join(s.split())

            def _segment_hits(segment_text: str, event_content: str) -> bool:
                norm_event = _norm(event_content)
                if _norm(segment_text) in norm_event:
                    return True
                return any(
                    _norm(s) and _norm(s) in norm_event
                    for s in segment_sentences(segment_text)
                )

            # 说话/互动对象归属（interlocutor）：按事件内容命中的子段取该段说话对象；
            # 无命中回退到批级。与 stored_marks 的段级匹配共用同一套 ``_segment_hits``。
            seg_interlocutors = {
                e.segment_id: (getattr(e, "interlocutor", None) or None)
                for e in valid
            }
            batch_interlocutor = next(
                (getattr(e, "interlocutor", None) for e in valid if getattr(e, "interlocutor", None)),
                None,
            )

            for ev in sealed:
                hits = [
                    e.segment_id for e in valid
                    if _segment_hits(e.text, ev.content_raw)
                ]
                for seg in hits:
                    result.stored_marks.setdefault(seg, []).append(ev.event_id)
                if not hits:
                    result.stored_marks.setdefault(
                        "+".join(e.segment_id for e in valid), [],
                    ).append(ev.event_id)

                # interlocutor：命中段里第一个有归属的段说话对象；无则回退批级；仍无则留 None。
                il = next((seg_interlocutors[seg] for seg in hits if seg_interlocutors.get(seg)), None)
                if il is None:
                    il = batch_interlocutor
                if il:
                    ev.interlocutor = il
                    try:
                        self.event_repo.save(ev)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("interlocutor save failed for %s: %s", ev.event_id, exc)

            if self.stored_marks_repo is not None:
                for seg, ids in result.stored_marks.items():
                    try:
                        self.stored_marks_repo.save(batch.subject_id, seg, ids)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "stored_marks save failed for %s: %s", seg, exc,
                        )
                        result.errors.append(f"stored_marks {seg}: {exc}")

        result.unclosed_count = len(self.meta_repo.get_unclosed_events())
        return result

    # ------------------------------------------------------------------
    # 回忆端口（design/1010）
    # ------------------------------------------------------------------

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        object_id: str | None = None,
        level: int = 1,
        limit: int | None = None,
        anchor_event_ids: tuple[str, ...] = (),
    ) -> tuple[RecalledFragment, ...]:
        """实现 MemoryBackendPort.recall（记忆恢复 + 预算管理 + 可选 recall_traces 落库）。"""
        if self.recall_pipeline is None:
            return ()
        fragments = self.recall_pipeline.recall(
            subject_id,
            query,
            object_id=object_id,
            level=level,
            limit=limit,
            anchor_event_ids=anchor_event_ids,
        )
        out = self.apply_recall_budget(subject_id, object_id, fragments)
        self._record_recall_trace(
            subject_id,
            query,
            object_id=object_id,
            level=level,
            limit=limit,
            anchor_event_ids=anchor_event_ids,
            fragments=out,
        )
        return out

    def apply_recall_budget(
        self,
        subject_id: str,
        object_id: str | None,
        fragments: tuple[RecalledFragment, ...],
    ) -> tuple[RecalledFragment, ...]:
        """把回忆条目按 6 级预算贪心分配；对话人（object_id）的人物肖像作为条目同场。

        - 未启用预算（``recall_budget=None``）→ 原样返回；
        - 启用时：肖像条目（若有）优先，与事件条目一起按评分排序、分档、截断到各自档位字数；
        - 总字数 ≤ ``recall_budget_chars``（0 => physical_redline）；超限尾部条目丢弃。
        """
        if self.recall_budget is None:
            return fragments
        items: list[RecalledFragment] = list(fragments)
        if (
            self.config.recall_budget_include_portrait
            and object_id
            and self.portrait_service is not None
        ):
            pfrag = self._make_portrait_fragment(subject_id, object_id)
            if pfrag is not None:
                items = [pfrag] + items
        if not items:
            return ()
        selected = self.recall_budget.allocate(items)
        out: list[RecalledFragment] = []
        for s in selected:
            frag = s["fragment"]
            used = s.get("used") or s["budget"]
            content = (frag.content or frag.text or "")[:used]
            out.append(frag.model_copy(update={"content": content, "summary_level": s["level"]}))
        return tuple(out)

    def assemble_recall_block(
        self,
        subject_id: str,
        query: str,
        *,
        object_id: str | None = None,
        level: int = 1,
        limit: int | None = None,
        anchor_event_ids: tuple[str, ...] = (),
        budget: int | None = None,
    ) -> dict:
        """预算受限的回忆块：条目 + 各级文本 + 总长（供上层组装上下文）。"""
        if self.recall_pipeline is None:
            return {"items": [], "total_length": 0, "budget": budget}
        fragments = self.recall_pipeline.recall(
            subject_id, query, object_id=object_id, level=level, limit=limit, anchor_event_ids=anchor_event_ids,
        )
        items: list[RecalledFragment] = list(fragments)
        if (
            self.config.recall_budget_include_portrait
            and object_id and self.portrait_service is not None
        ):
            pfrag = self._make_portrait_fragment(subject_id, object_id)
            if pfrag is not None:
                items = [pfrag] + items
        b = budget or (self.recall_budget.budget_chars() if self.recall_budget else None)
        selected = self.recall_budget.allocate(items, budget=b) if self.recall_budget else [
            {"fragment": it, "level": it.summary_level or "L1", "budget": len(it.content or it.text or "")}
            for it in items
        ]
        out_items = []
        total = 0
        for s in selected:
            frag = s["fragment"]
            used = s.get("used") or s["budget"]
            content = (frag.content or frag.text or "")[: used]
            total += len(content)
            out_items.append({
                "event_id": frag.event_id, "object_id": frag.object_id, "interlocutor": frag.interlocutor,
                "kind": frag.kind, "level": s["level"], "budget": s["budget"], "used": used,
                "content": content, "score": frag.score,
            })
        return {"items": out_items, "total_length": total, "budget": b}

    def _make_portrait_fragment(self, subject_id: str, object_id: str) -> RecalledFragment | None:
        """把对话人的人物肖像包装成回忆条目（最高优先）。"""
        p = self.portrait_service.get_portrait(object_id) if self.portrait_service else None
        if p is None:
            return None
        text = p.longest_level.text if p.longest_level else ""
        if not text:
            return None
        return RecalledFragment(
            event_id=f"portrait:{object_id}",
            text=text,
            content=text,
            kind="portrait",
            object_id=object_id,
            interlocutor=object_id,
            # 对话人的肖像=核心上下文，优先级高于普通事件条目，保证在预算内优先纳入。
            score=100.0,
            summary_level=p.max_level,
            occurred_at=p.updated_at,
        )

    def _record_recall_trace(
        self,
        subject_id: str,
        query: str,
        *,
        object_id: str | None,
        level: int,
        limit: int | None,
        anchor_event_ids: tuple[str, ...],
        fragments: tuple[RecalledFragment, ...],
    ) -> None:
        """把本次 recall 的输入与命中条目写入 recall_traces（observability）。

        只作观测，不影响召回结果；关闭 ``recall_trace_enabled`` 或未装配 repo 时静默跳过。
        """
        if not self.config.recall_trace_enabled or self.recall_trace_repo is None:
            return
        try:
            items = [
                RecallTraceItem(
                    event_id=f.event_id,
                    object_id=f.object_id,
                    interlocutor=f.interlocutor,
                    score=f.score,
                    summary_level=f.summary_level,
                    source_ids=list(f.source_ids or []),
                    text=f.text or "",
                    content=f.content or "",
                    occurred_at=f.occurred_at,
                )
                for f in fragments
            ]
            max_items = int(getattr(self.config, "recall_trace_max_items", 0) or 0)
            if max_items > 0 and len(items) > max_items:
                items = items[:max_items]
            trace = RecallTrace(
                recall_id=uuid.uuid4().hex,
                subject_id=subject_id,
                query=query,
                object_id=object_id,
                level=level,
                limit=limit,
                anchor_event_ids=list(anchor_event_ids or ()),
                items=items,
            )
            self.recall_trace_repo.save(trace)
        except Exception as exc:  # noqa: BLE001
            # 观测失败不影响召回结果本身
            logger.warning("recall_trace save failed: %s", exc)

    # ------------------------------------------------------------------
    # 人物肖像（design/1010 §8.4 portrait）
    # ------------------------------------------------------------------

    def portrait(self, subject_id: str, object_id: str) -> dict | None:
        """返回某对象的人物肖像（10 级渐进摘要的最长级 + 概要）。"""
        if self.portrait_service is None:
            return None
        p = self.portrait_service.get_portrait(object_id)
        if p is None:
            return None
        longest = p.longest_level
        return {
            "subject_id": p.subject_id,
            "object_id": p.object_id,
            "name": p.name,
            "max_level": p.max_level,
            "total_content_len": p.total_content_len,
            "compression_ratio": p.compression_ratio,
            "fatigue": p.fatigue,
            "visible_summary": longest.text if longest else "",
            "levels": {k: v.text for k, v in p.levels.items()},
            "updated_at": p.updated_at,
        }

    # ------------------------------------------------------------------
    # 便捷方法（API 用）
    # ------------------------------------------------------------------

    def query_role(self, name_or_id: str) -> dict:
        role = self.role_repo.get(name_or_id) or self.role_service.find_role(name_or_id)
        if not role:
            return {"error": f"Role '{name_or_id}' not found"}
        summary = self.role_service.get_white_painting_summary(role.role_id)
        return {
            "role": role.model_dump(mode="json", exclude={"white_painting"}),
            "white_painting_summary": summary,
        }

    def tombstone(self, event_id: str, reason: str, replacement_id: str | None = None) -> bool:
        return self.belief_revision_service.tombstone_event(
            event_id, reason=reason, replacement_event_id=replacement_id
        )
