"""回忆检索管线：两路材料、原始检索信号、对象角色加权与预算选级。

对象角色权重参与排序分；可及性只在加权分接近时调整顺序。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..models.event import Event
from ..models.interlocutor import InterlocutorAttribution
from ..port import RecalledFragment
from ..storage.repository import EventRepository, ObjectTimelineRepository
from ..utils.zh_normalize import normalize_for_substring_match as _norm
from .intent import RuleIntentClassifier

logger = logging.getLogger(__name__)


_IMPORTANCE_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
_INTERLOCUTOR_OBJECT_WEIGHT = 0.7
_INVOLVED_OBJECT_WEIGHT = 0.3
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:℃|°c|°|%)?", re.IGNORECASE)
_LATIN_RE = re.compile(r"[a-z][a-z0-9+\-]{1,}")


def _shown_object_id(ev: Event, wanted: tuple[str, ...]) -> str | None:
    """结果上的对象编号取事件自己的对象；调用方点名的对象优先。"""
    if wanted:
        historical_interlocutor_ids = {
            item.object_id for item in ev.interlocutor_attributions if item.object_id
        }
        for oid in wanted:
            if (
                oid in historical_interlocutor_ids
                or any((not role.is_subject) and role.role_id == oid for role in ev.role_list)
            ):
                return oid
    return _event_object_id(ev)


def _event_object_id(ev: Event) -> str | None:
    """取这条记忆事件自己的对象（外部角色），不做成查询对象。"""
    others = [r for r in ev.role_list if not r.is_subject]
    if not others:
        return None
    others.sort(key=lambda r: _IMPORTANCE_ORDER.get(r.importance.value, 99))
    return others[0].role_id


def retrieval_text(event: Event) -> str:
    """事件向量文本：只有 L1。没有 L1 时返回空串，不改用原文。"""
    return _level_text(event.summaries, "L1")


def material_source_hash(text: str) -> str:
    """L1 正文的哈希。正文不变时，索引不再重新嵌入。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _level_text(levels: dict | None, key: str) -> str:
    raw = (levels or {}).get(key) or ""
    return raw.strip() if isinstance(raw, str) else ""


@dataclass
class _Candidate:
    """一条候选材料。原始相关度取语义和词面的较大值，再乘对象角色权重排序。"""

    material_type: str
    event_id: str
    object_id: str | None
    semantic: float
    lexical: float
    object_hit: float
    temporal: float | None
    accessibility: float
    levels: dict[str, str]
    raw: str
    kind: str
    source_ids: list[str] = field(default_factory=list)
    occurred_at: datetime | None = None
    event: Event | None = None
    object_role_weight: float = 1.0
    interlocutor_attributions: list[InterlocutorAttribution] = field(default_factory=list)

    @property
    def relevance(self) -> float:
        return max(float(self.semantic), float(self.lexical))

    @property
    def rank_score(self) -> float:
        return self.relevance * self.object_role_weight


def order_by_relevance(items: list[_Candidate], close: float) -> list[_Candidate]:
    """加权分明显不同时保持顺序；只有落在接近幅度内，可及性才改序。"""
    if not items:
        return []
    ordered = sorted(items, key=lambda c: (c.rank_score, c.accessibility), reverse=True)
    if close <= 0 or len(ordered) < 2:
        return ordered
    out: list[_Candidate] = []
    cluster = [ordered[0]]
    anchor = ordered[0].rank_score

    def _flush(group: list[_Candidate]) -> None:
        group.sort(key=lambda c: (c.accessibility, c.rank_score), reverse=True)
        out.extend(group)

    for cand in ordered[1:]:
        if anchor - cand.rank_score <= close:
            cluster.append(cand)
        else:
            _flush(cluster)
            cluster = [cand]
            anchor = cand.rank_score
    _flush(cluster)
    return out


def lexical_score(query: str, text: str) -> float:
    """词面分。专名、数字、完整短语命中为 1；其余按最长公共子串占查询的比例。"""
    q = _norm(query)
    t = _norm(text)
    if not q or not t:
        return 0.0
    if len(q) >= 2 and q in t:
        return 1.0
    for match in _NUMBER_RE.finditer(q):
        tok = match.group(0)
        if len(tok) >= 1 and tok in t:
            return 1.0
    for match in _LATIN_RE.finditer(q):
        tok = match.group(0)
        if tok in t:
            return 1.0
    q2 = q[:80]
    found = 0
    for length in range(len(q2), 1, -1):
        hit = False
        for i in range(0, len(q2) - length + 1):
            if q2[i : i + length] in t:
                found = length
                hit = True
                break
        if hit:
            break
    if found >= 2:
        return found / len(q2)
    return _bigram_jaccard(_bigrams(q), _bigrams(t))


def _bigrams(text: str) -> set[str]:
    t = _norm(text)
    return {t[i : i + 2] for i in range(max(0, len(t) - 1))}


def _bigram_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _level_index(key: str) -> int:
    if key.startswith("L") and key[1:].isdigit():
        return int(key[1:])
    return 99


def select_representation(
    levels: dict[str, str],
    raw: str,
    budget: int,
    *,
    expand_raw: bool,
) -> tuple[str, str] | None:
    """选取放得进预算且信息最多的完整一级。放不下则不返回，也不截断原文。"""
    if budget < 0:
        return None
    if expand_raw and raw and len(raw) <= budget:
        return raw, "raw"
    options = [
        (len(text), _level_index(key), key, text)
        for key, text in (levels or {}).items()
        if text and text.strip()
    ]
    options.sort(key=lambda item: (-item[0], item[1]))
    for length, _idx, key, text in options:
        if length <= budget:
            return text, key
    return None


class RecallPipeline:
    """语义与词面入候选，对象和时间只在调用方传入时限制集合。"""

    def __init__(
        self,
        embedding,
        vector_store,
        event_repo: EventRepository,
        object_timeline_repo: ObjectTimelineRepository | None = None,
        *,
        reranker=None,
        rrf_k: int = 60,
        top_k: int = 60,
        rerank_top_n: int = 20,
        half_life_days: float = 60.0,
        silence_threshold: float = 0.02,
        reinforce_multiplier: float = 1.5,
        reinforce_cap: float = 300.0,
        intent_classifier: RuleIntentClassifier | None = None,
        recency_enabled: bool = False,
        recency_window_events: int = 80,
        recency_top_k: int = 40,
        factor_alpha: float = 0.5,
        mood_beta: float = 0.2,
        event_fatigue: float = 1.0,
        meta_repo=None,
        include_unclosed: bool = True,
        unclosed_same_object_score: float = 90.0,
        role_repo=None,
        semantic_min: float = 0.35,
        lexical_min: float = 0.05,
        relevance_close: float = 0.08,
        default_budget_chars: int = 8000,
        embedding_model: str = "hash",
        embedding_version: int = 1,
    ):
        self._embedding = embedding
        self._vector_store = vector_store
        self._event_repo = event_repo
        self._object_timeline_repo = object_timeline_repo
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n
        self._half_life_days = half_life_days
        self._silence_threshold = silence_threshold
        self._reinforce_multiplier = reinforce_multiplier
        self._reinforce_cap = reinforce_cap
        self._intent_classifier = intent_classifier
        self._recency_enabled = recency_enabled
        self._recency_window_events = recency_window_events
        self._recency_top_k = recency_top_k
        self._factor_alpha = factor_alpha
        self._mood_beta = mood_beta
        # 事件记忆疲态：与遗忘率协同（乘进 effective factor）；<1 时更快"忘记"。
        self._event_fatigue = event_fatigue
        self._meta_repo = meta_repo
        self._include_unclosed = include_unclosed
        self._unclosed_same_object_score = unclosed_same_object_score
        self._role_repo = role_repo
        self._semantic_min = semantic_min
        self._lexical_min = lexical_min
        self._relevance_close = relevance_close
        self._default_budget_chars = default_budget_chars
        self._embedding_model = embedding_model
        self._embedding_version = int(embedding_version)

    # ------------------------------------------------------------------
    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        object_id: str | None = None,
        object_ids: tuple[str, ...] | None = None,
        interlocutor_object_id: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        budget_chars: int | None = None,
        expand_raw: bool = False,
        level: int = 1,
        limit: int | None = None,
        anchor_event_ids: tuple[str, ...] = (),
        channels: tuple[str, ...] | None = None,
        reinforce: bool = True,
    ) -> tuple[RecalledFragment, ...]:
        """交回事件摘要和对象事件事实；对象角色权重只参与排序。"""
        del level, anchor_event_ids  # 档位由预算决定；锚点不作为这一版的检索路
        q = (query or "").strip()
        if not _norm(q):
            return ()

        wanted = self._wanted_objects(object_id, object_ids, interlocutor_object_id)
        use_semantic = channels is None or "semantic" in channels
        use_lexical = channels is None or "lexical" in channels
        budget = self._default_budget_chars if budget_chars is None else int(budget_chars)
        now = datetime.now()

        events = [
            ev for ev in self._candidate_events(subject_id)
            if self._in_time_range(ev, time_range) and self._matches_objects(subject_id, ev, wanted)
        ]
        semantic = (
            self._semantic_signals(subject_id, q, wanted, time_range) if use_semantic else {}
        )
        candidates: list[_Candidate] = []
        for ev in events:
            candidates.extend(self._event_candidates(
                ev, q, wanted, semantic, time_range, now,
                interlocutor_object_id=interlocutor_object_id,
                use_semantic=use_semantic, use_lexical=use_lexical,
            ))
        candidates.extend(self._unclosed_candidates(
            subject_id, q, wanted, time_range,
            interlocutor_object_id=interlocutor_object_id,
            use_lexical=use_lexical,
        ))
        admitted = [c for c in candidates if self._admitted(c, use_semantic, use_lexical)]
        ordered = self._arrange(admitted, wanted)
        fragments = self._fill_budget(ordered, budget, expand_raw=expand_raw, limit=limit)
        if reinforce:
            seen: set[str] = set()
            for frag in fragments:
                if frag.type == "unclosed" or frag.event_id in seen:
                    continue
                seen.add(frag.event_id)
                ev = self._event_repo.get(frag.event_id)
                if ev is not None:
                    self._reinforce(ev)
        return tuple(fragments)

    def index_event(self, event: Event) -> None:
        """为事件 L1 和每条对象事实 L1 写向量。正文与版本都没变则跳过。"""
        object_ids = [role.role_id for role in event.role_list if not role.is_subject]
        event_object_ids = list(dict.fromkeys((
            *object_ids,
            *(item.object_id for item in event.interlocutor_attributions if item.object_id),
        )))
        pending: list[tuple[str, str, dict]] = []
        l1 = retrieval_text(event)
        if l1:
            payload = self._index_payload(
                event, material_type="event", object_id=None,
                object_ids=event_object_ids, text=l1,
            )
            if not self._index_unchanged(event.event_id, payload):
                pending.append((event.event_id, l1, payload))
        else:
            self._vector_store.delete(event.event_id)
        if self._role_repo is not None:
            for role in event.role_list:
                if role.is_subject:
                    continue
                logical_id = f"{event.event_id}#fact#{role.role_id}"
                fact = self._fact_l1(role.role_id, event.event_id)
                if not fact:
                    self._vector_store.delete(logical_id)
                    continue
                payload = self._index_payload(
                    event, material_type="object_event_fact", object_id=role.role_id,
                    object_ids=[role.role_id], text=fact,
                )
                if not self._index_unchanged(logical_id, payload):
                    pending.append((logical_id, fact, payload))
        if not pending:
            return
        vectors = self._embedding.embed_documents([text for _pid, text, _payload in pending])
        for (logical_id, _text, payload), vector in zip(pending, vectors):
            self._vector_store.upsert(logical_id, vector, payload)

    def _index_payload(
        self,
        event: Event,
        *,
        material_type: str,
        object_id: str | None,
        object_ids: list[str],
        text: str,
    ) -> dict:
        moment = event.occurred_at or event.create_time
        return {
            "type": material_type,
            "subject_id": event.subject_id,
            "event_id": event.event_id,
            "object_id": object_id,
            "object_ids": list(object_ids),
            "representation_level": "L1",
            "source_hash": material_source_hash(text),
            "embedding_model": self._embedding_model,
            "embedding_version": self._embedding_version,
            "occurred_at": moment.timestamp() if moment else None,
        }

    def _index_unchanged(self, logical_id: str, payload: dict) -> bool:
        getter = getattr(self._vector_store, "get", None)
        if getter is None:
            return False
        existing = getter(logical_id)
        if not existing:
            return False
        return (
            existing.get("source_hash") == payload["source_hash"]
            and existing.get("embedding_model") == payload["embedding_model"]
            and existing.get("embedding_version") == payload["embedding_version"]
            and existing.get("object_id") == payload.get("object_id")
            and existing.get("object_ids") == payload.get("object_ids")
        )

    def _wanted_objects(
        self,
        object_id: str | None,
        object_ids: tuple[str, ...] | None,
        interlocutor_object_id: str | None = None,
    ) -> tuple[str, ...]:
        found: list[str] = []
        if interlocutor_object_id:
            found.append(interlocutor_object_id)
        for oid in object_ids or ():
            if oid and oid not in found:
                found.append(oid)
        if object_id and object_id not in found:
            found.append(object_id)
        return tuple(found)

    def _in_time_range(self, ev: Event, time_range: tuple[datetime, datetime] | None) -> bool:
        if time_range is None:
            return True
        start, end = time_range
        moment = ev.occurred_at or ev.create_time
        if moment is None:
            return False
        return start <= moment <= end

    def _matches_objects(self, subject_id: str, ev: Event, wanted: tuple[str, ...]) -> bool:
        if not wanted:
            return True
        if any(
            item.object_id in wanted
            for item in ev.interlocutor_attributions
            if item.object_id
        ):
            return True
        if any((not role.is_subject) and role.role_id in wanted for role in ev.role_list):
            return True
        route = self._object_route(subject_id, wanted[0], [ev])
        if len(wanted) == 1:
            return ev.event_id in route
        if self._object_timeline_repo is None:
            return False
        for oid in wanted:
            entries = self._object_timeline_repo.list_by_object(subject_id, oid)
            if any(entry.event_id == ev.event_id for entry in entries):
                return True
        return False

    def _semantic_signals(
        self,
        subject_id: str,
        query: str,
        wanted: tuple[str, ...],
        time_range: tuple[datetime, datetime] | None,
    ) -> dict[tuple[str, str, str], float]:
        vector = self._embedding.embed_query(query)
        ranges = None
        if time_range is not None:
            start, end = time_range
            ranges = {"occurred_at": (start.timestamp(), end.timestamp())}
        hits = self._vector_store.search(
            vector,
            top_k=self._top_k,
            payload_filter={
                "subject_id": subject_id,
                "embedding_model": self._embedding_model,
                "embedding_version": self._embedding_version,
            },
            any_of={"object_ids": list(wanted)} if wanted else None,
            ranges=ranges,
        )
        scores: dict[tuple[str, str, str], float] = {}
        for hit in hits:
            payload = hit.get("payload") or {}
            event_id = payload.get("event_id") or hit.get("event_id")
            if not event_id:
                continue
            material = payload.get("type") or "event"
            object_id = payload.get("object_id") or ""
            if material != "object_event_fact":
                material = "event"
                object_id = ""
            key = (material, str(event_id), str(object_id))
            score = float(hit.get("score") or 0.0)
            if score > scores.get(key, 0.0):
                scores[key] = score
        return scores

    def _event_candidates(
        self,
        ev: Event,
        query: str,
        wanted: tuple[str, ...],
        semantic: dict[tuple[str, str, str], float],
        time_range: tuple[datetime, datetime] | None,
        now: datetime,
        *,
        interlocutor_object_id: str | None,
        use_semantic: bool,
        use_lexical: bool,
    ) -> list[_Candidate]:
        access = self._effective_factor(ev, now)
        temporal = 1.0 if time_range is not None else None
        object_hit = 1.0 if wanted else 0.0
        historical_interlocutor_ids = {
            item.object_id for item in ev.interlocutor_attributions if item.object_id
        }
        event_object_role_weight = (
            1.0 if not wanted else
            _INTERLOCUTOR_OBJECT_WEIGHT
            if interlocutor_object_id and interlocutor_object_id in historical_interlocutor_ids
            else _INVOLVED_OBJECT_WEIGHT
        )
        levels = {
            str(key): text.strip()
            for key, text in (ev.summaries or {}).items()
            if text and str(text).strip()
        }
        event_text = "\n".join(levels.values())
        out = [_Candidate(
            material_type="event",
            event_id=ev.event_id,
            object_id=_shown_object_id(ev, wanted),
            semantic=semantic.get(("event", ev.event_id, ""), 0.0) if use_semantic else 0.0,
            lexical=lexical_score(query, event_text) if use_lexical else 0.0,
            object_hit=object_hit,
            temporal=temporal,
            accessibility=access,
            levels=levels,
            raw=ev.content_raw or "",
            kind=ev.origin,
            source_ids=list(ev.source_ids or []),
            occurred_at=ev.occurred_at or ev.create_time,
            event=ev,
            object_role_weight=event_object_role_weight,
        )]
        for role in ev.role_list:
            if role.is_subject:
                continue
            if wanted and role.role_id not in wanted:
                continue
            fact_levels = self._fact_levels(role.role_id, ev.event_id)
            if not fact_levels:
                continue
            fact_text = "\n".join(fact_levels.values())
            out.append(_Candidate(
                material_type="object_event_fact",
                event_id=ev.event_id,
                object_id=role.role_id,
                semantic=semantic.get(("object_event_fact", ev.event_id, role.role_id), 0.0) if use_semantic else 0.0,
                lexical=lexical_score(query, fact_text) if use_lexical else 0.0,
                object_hit=object_hit,
                temporal=temporal,
                accessibility=access,
                levels=fact_levels,
                raw=fact_levels.get("L1") or fact_text,
                kind="object_event_fact",
                source_ids=list(ev.source_ids or []),
                occurred_at=ev.occurred_at or ev.create_time,
                event=ev,
                object_role_weight=(
                    1.0 if not wanted else
                    _INTERLOCUTOR_OBJECT_WEIGHT
                    if interlocutor_object_id
                    and interlocutor_object_id == role.role_id
                    and interlocutor_object_id in historical_interlocutor_ids
                    else _INVOLVED_OBJECT_WEIGHT
                ),
            ))
        return out

    def _unclosed_candidates(
        self,
        subject_id: str,
        query: str,
        wanted: tuple[str, ...],
        time_range: tuple[datetime, datetime] | None,
        *,
        interlocutor_object_id: str | None,
        use_lexical: bool,
    ) -> list[_Candidate]:
        if not self._include_unclosed or self._meta_repo is None or not use_lexical:
            return []
        try:
            items = self._meta_repo.get_unclosed_events()
        except Exception:  # noqa: BLE001
            logger.debug("list unclosed for recall failed", exc_info=True)
            return []
        out: list[_Candidate] = []
        for item in items:
            sid = getattr(item, "subject_id", "") or ""
            if sid and sid != subject_id:
                continue
            text = (item.merged_content or "").strip()
            if not text:
                continue
            roles = list(getattr(item, "identified_roles", None) or [])
            interlocutor_attributions = list(
                getattr(item, "interlocutor_attributions", None) or []
            )
            historical_interlocutor_ids = {
                attr.object_id for attr in interlocutor_attributions if attr.object_id
            }
            roles = list(dict.fromkeys((*roles, *historical_interlocutor_ids)))
            if wanted and not any(oid in roles for oid in wanted):
                continue
            moment = getattr(item, "last_hit_time", None) or getattr(item, "updated_at", None)
            if time_range is not None:
                start, end = time_range
                if moment is None or not (start <= moment <= end):
                    continue
            out.append(_Candidate(
                material_type="unclosed",
                event_id=item.id,
                object_id=next((oid for oid in wanted if oid in roles), None),
                semantic=0.0,
                lexical=lexical_score(query, text),
                object_hit=1.0 if wanted else 0.0,
                temporal=1.0 if time_range is not None else None,
                accessibility=1.0,
                levels={},
                raw=text,
                kind="unclosed",
                occurred_at=moment,
                object_role_weight=(
                    1.0 if not wanted else
                    _INTERLOCUTOR_OBJECT_WEIGHT
                    if interlocutor_object_id
                    and interlocutor_object_id in historical_interlocutor_ids
                    else _INVOLVED_OBJECT_WEIGHT
                ),
                interlocutor_attributions=interlocutor_attributions,
            ))
        return out

    def _admitted(self, cand: _Candidate, use_semantic: bool, use_lexical: bool) -> bool:
        if cand.material_type == "unclosed":
            return cand.lexical >= self._lexical_min
        semantic_ok = use_semantic and cand.semantic >= self._semantic_min
        lexical_ok = use_lexical and cand.lexical >= self._lexical_min
        return semantic_ok or lexical_ok

    def _arrange(self, admitted: list[_Candidate], wanted: tuple[str, ...]) -> list[_Candidate]:
        if not wanted:
            return order_by_relevance(admitted, self._relevance_close)
        facts = [c for c in admitted if c.material_type == "object_event_fact"]
        rest = [c for c in admitted if c.material_type != "object_event_fact"]
        return [
            *order_by_relevance(facts, self._relevance_close),
            *order_by_relevance(rest, self._relevance_close),
        ]

    def _fill_budget(
        self,
        ordered: list[_Candidate],
        budget: int,
        *,
        expand_raw: bool,
        limit: int | None,
    ) -> list[RecalledFragment]:
        remaining = budget
        seen: set[tuple[str, str, str]] = set()
        out: list[RecalledFragment] = []
        for cand in ordered:
            if limit is not None and len(out) >= limit:
                break
            key = (cand.material_type, cand.event_id, cand.object_id or "")
            if key in seen:
                continue
            seen.add(key)
            if cand.material_type == "unclosed":
                picked = (cand.raw, "raw") if len(cand.raw) <= remaining else None
            else:
                picked = select_representation(
                    cand.levels, cand.raw, remaining, expand_raw=expand_raw,
                )
            if picked is None:
                continue
            content, level_name = picked
            remaining -= len(content)
            signals = {
                "semantic": round(cand.semantic, 6),
                "lexical": round(cand.lexical, 6),
                "object": cand.object_hit,
            }
            if cand.object_hit:
                signals["object_role_weight"] = cand.object_role_weight
            if cand.temporal is not None:
                signals["temporal"] = cand.temporal
            text = cand.raw if cand.material_type == "event" else (cand.levels.get("L1") or cand.raw)
            attributions = cand.interlocutor_attributions or (
                cand.event.interlocutor_attributions if cand.event is not None else []
            )
            interlocutor_ids = list(dict.fromkeys(
                item.object_id for item in attributions if item.object_id
            ))
            out.append(RecalledFragment(
                event_id=cand.event_id,
                text=text,
                content=content,
                type=cand.material_type,
                kind=cand.kind,
                object_id=cand.object_id,
                interlocutor=interlocutor_ids[0] if len(interlocutor_ids) == 1 else None,
                interlocutor_object_ids=interlocutor_ids,
                interlocutor_attributions=attributions,
                source_ids=cand.source_ids,
                score=round(cand.rank_score, 6),
                summary_level=level_name,
                representation_level=level_name,
                signals=signals,
                occurred_at=cand.occurred_at,
            ))
        return out

    def _fact_l1(self, role_id: str, event_id: str) -> str:
        levels = self._fact_levels(role_id, event_id)
        return levels.get("L1") or ""

    def _fact_levels(self, role_id: str, event_id: str) -> dict[str, str]:
        if self._role_repo is None:
            return {}
        entry = self._role_repo.get_white_painting_by_event(role_id, event_id)
        if entry is None:
            return {}
        levels: dict[str, str] = {}
        l1 = (entry.l1_mention or entry.role_summary or "").strip()
        if l1:
            levels["L1"] = l1
        if entry.l2_interaction and entry.l2_interaction.strip():
            levels["L2"] = entry.l2_interaction.strip()
        if entry.l3_decision and entry.l3_decision.strip():
            levels["L3"] = entry.l3_decision.strip()
        return levels

    # ------------------------------------------------------------------
    def _candidate_events(self, subject_id: str) -> list[Event]:
        return [
            e for e in self._event_repo.list_all(exclude_tombstoned=True)
            if (e.subject_id or "") == subject_id and not e.is_tombstoned
        ]

    def _retrieval_text(self, ev: Event) -> str:
        return retrieval_text(ev)

    def _object_route(
        self, subject_id: str, object_id: str | None, events: list[Event] | None = None
    ) -> dict[str, int]:
        """Object route: resolve from candidate events' role_list first (memory mode
        main path after white-painting migration), fall back to the legacy
        object_timeline table for role-system era data."""
        if not object_id:
            return {}
        if events is None:
            events = self._candidate_events(subject_id)
        hits = [ev for ev in events if any(r.role_id == object_id for r in ev.role_list)]
        if hits:
            return {ev.event_id: idx + 1 for idx, ev in enumerate(hits)}
        if self._object_timeline_repo is not None:
            entries = self._object_timeline_repo.list_by_object(subject_id, object_id)
            if entries:
                return {e.event_id: idx + 1 for idx, e in enumerate(entries)}
        return {}

    def _recency_route(self, events: list[Event]) -> dict[str, int]:
        """近因通道：窗口内按 create_time 降序给 RRF 排名（design/1010 §3）。"""
        if not events:
            return {}
        window = max(1, self._recency_window_events)
        k = max(1, self._recency_top_k)
        ordered = sorted(events, key=lambda ev: ev.create_time, reverse=True)[:window]
        return {ev.event_id: idx + 1 for idx, ev in enumerate(ordered[:k])}

    def _effective_factor(self, ev: Event, now: datetime) -> float:
        age_days = max((now - ev.create_time).total_seconds() / 86400.0, 0.0)
        decay = 0.5 ** (age_days / self._half_life_days)
        # 记忆疲态与遗忘率协同：effective = event_fatigue × forgetting_factor × 时间衰减。
        return self._event_fatigue * float(getattr(ev, "forgetting_factor", 1.0) or 1.0) * decay

    def _reinforce(self, ev: Event) -> None:
        factor = min(
            float(getattr(ev, "forgetting_factor", 1.0) or 1.0) * self._reinforce_multiplier,
            self._reinforce_cap,
        )
        self._event_repo.update_forgetting_factor(ev.event_id, factor)
