"""回忆检索管线：小库全量语义检索、大库近期与长期分池、预算选级。

可及性概率仅决定长期事件是否进入语义检索；不修改向量相关度分数。
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
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


def _shown_object_id(ev: Event, wanted: tuple[str, ...]) -> str | None:
    """结果上的对象编号取事件自己的对象；调用方点名的对象优先。"""
    if wanted:
        historical_interlocutor_ids = {
            item.object_id for item in ev.interlocutor_attributions if item.object_id
        }
        for oid in wanted:
            if (
                oid in historical_interlocutor_ids
                or oid in ev.refinement_objects.values()
                or any((not role.is_subject) and role.role_id == oid for role in ev.role_list)
            ):
                return oid
    return _event_object_id(ev)


def _event_object_id(ev: Event) -> str | None:
    """取这条记忆事件自己的对象（外部角色），不做成查询对象。"""
    others = [r for r in ev.role_list if not r.is_subject]
    if not others:
        return next(iter(ev.refinement_objects.values()), None)
    others.sort(key=lambda r: _IMPORTANCE_ORDER.get(r.importance.value, 99))
    return others[0].role_id


def retrieval_text(event: Event, level: str = "L1") -> str:
    """索引来源与取回表示独立；不混用两个版本的向量。"""
    return (event.content_raw or "").strip() if level == "L0" else _level_text(event.summaries, "L1")


def material_source_hash(text: str) -> str:
    """L1 正文的哈希。正文不变时，索引不再重新嵌入。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _level_text(levels: dict | None, key: str) -> str:
    raw = (levels or {}).get(key) or ""
    return raw.strip() if isinstance(raw, str) else ""


@dataclass
class _Candidate:
    """一条候选材料；语义与词面一致时加分，再乘对象角色权重。"""

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
    is_recent_lane: bool = False
    detail_scores: dict[str, float] | None = None

    @property
    def relevance(self) -> float:
        semantic = max(0.0, float(self.semantic))
        lexical = max(0.0, float(self.lexical))
        # Agreement between independent signals is evidence in its own right.
        # Keep single-route results at their original score.
        return min(1.0, max(semantic, lexical) + 0.25 * min(semantic, lexical))

    @property
    def rank_score(self) -> float:
        return self.relevance * self.object_role_weight


@dataclass
class RecallCandidate:
    """一次 recall 中候选材料的去向（仅诊断；不参与排序、不占预算、不影响结果）。

    ``_fill_budget`` 按最终顺序访问候选，选中的与被淘汰的都记一条，因此可以离线回答
    「正确材料排第几、有多大、为什么没交付」。``reason`` 取 ``selected`` / ``limit`` /
    ``duplicate`` / ``no_room`` 四值：``limit`` 是条数上限截断，``duplicate`` 是同一事件
    的同一材料类型重复占位，``no_room`` 是剩余预算放不下任何一级表示。
    """

    call: int
    rank: int
    material_type: str
    event_id: str
    object_id: str | None
    score: float
    selected: bool
    reason: str
    representation_level: str | None = None
    chars: int = 0
    budget_remaining: int = 0
    similarity: float | None = None

    def as_dict(self) -> dict:
        return {
            "call": self.call,
            "rank": self.rank,
            "material_type": self.material_type,
            "event_id": self.event_id,
            "object_id": self.object_id,
            "score": self.score,
            "selected": self.selected,
            "reason": self.reason,
            "representation_level": self.representation_level,
            "chars": self.chars,
            "budget_remaining": self.budget_remaining,
            "similarity": self.similarity,
        }


CANDIDATE_REASONS = (
    "selected", "limit", "duplicate", "no_room",
    "detail_attached", "detail_no_room", "detail_limit",
    "detail_below_threshold",
)


def source_date_anchor(dates: list[str], locale: str = "zh") -> str:
    """同一条结果里的日期锚点使用同一种写法。相对日期以来源日期为准。"""
    shown = [value[:10] for value in dates if value]
    if not shown:
        return ""
    if locale == "en":
        if len(shown) == 1:
            return f"[Source statement date: {shown[0]}; relative dates refer to this date.] "
        return (
            f"[Source statements span {shown[0]} to {shown[-1]}; "
            "relative-date anchor is ambiguous.] "
        )
    if len(shown) == 1:
        return f"[来源日期：{shown[0]}；文中的相对日期以这一天为准。] "
    return f"[来源日期从 {shown[0]} 到 {shown[-1]}；相对日期不能折算到同一天。] "


def query_terms_for_detail(query: str) -> tuple[str, set[str]]:
    """查询词：纯英文走现有词面词；否则按标点切开，至少两个字。"""
    if re.search(r"[A-Za-z]", query) and not re.search(r"[\u3400-\u9fff]", query):
        return "en", _english_terms(query)
    return "text", {piece.casefold() for piece in _ENGLISH_WORDS.findall(query) if len(piece) >= 2}


def terms_present(kind: str, terms: set[str], text: str) -> set[str]:
    if not terms or not text:
        return set()
    if kind == "en":
        return terms & _english_terms(text)
    folded = text.casefold()
    return {term for term in terms if term in folded}


def missing_query_terms(query: str, summary: str, detail: str) -> set[str]:
    """下层里有、已交付摘要里没有的查询词。没有这种词就不附细节。"""
    kind, terms = query_terms_for_detail(query)
    return terms_present(kind, terms, detail) - terms_present(kind, terms, summary)


def order_by_relevance(items: list[_Candidate], close: float) -> list[_Candidate]:
    """加权分明显不同时保持顺序；只有落在接近幅度内，可及性才改序。"""
    if not items:
        return []
    ordered = sorted(
        items,
        key=lambda c: (
            c.rank_score,
            c.material_type == "object_event_fact",
            c.accessibility,
        ),
        reverse=True,
    )
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


_ENGLISH_WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_ENGLISH_FUNCTION_WORDS = frozenset("""a an the and or but of to in on at by for from with
as is are was were be been being do does did have has had can could would should
will shall may might what which who whom whose where when why how i me my mine
you your yours he him his she her hers it its we us our ours they them their theirs
this that these those than then so not no yes s t m re ve ll d""".split())


def _english_terms(text: str) -> set[str]:
    result = set()
    for word in _ENGLISH_WORDS.findall(text.casefold()):
        word = word.replace("’", "'")
        if word.endswith("'s"):
            word = word[:-2]
        if word in _ENGLISH_FUNCTION_WORDS:
            continue
        # Small, dependency-free plural normalization; do not match substrings
        # such as 'art' inside 'party'. Semantic matching handles paraphrases.
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        result.add(word)
    return result


def english_word_coverage(query: str, text: str) -> float:
    terms = _english_terms(query)
    return len(terms & _english_terms(text)) / len(terms) if terms else 0.0


def lexical_score(query: str, text: str, *, english_words: bool = False) -> float:
    """词面分：完整命中、最长片段和查询双字片段覆盖率。"""
    if english_words and re.search(r"[A-Za-z]", query) and not re.search(r"[\u3400-\u9fff]", query):
        return english_word_coverage(query, text)
    q = "".join(ch for ch in _norm(query) if ch.isalnum() or ch in "℃°%+-")
    t = "".join(ch for ch in _norm(text) if ch.isalnum() or ch in "℃°%+-")
    if not q or not t:
        return 0.0
    if len(q) >= 2 and q in t:
        return 1.0
    # A name or date shared by many memories cannot by itself prove that the
    # rest of a longer question matches. In particular "lux" previously gave
    # every lux fact the maximum lexical score, burying the requested topic.
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
    query_bigrams = _bigrams(q)
    text_bigrams = _bigrams(t)
    coverage = (
        len(query_bigrams & text_bigrams) / len(query_bigrams)
        if query_bigrams else 0.0
    )
    return max(
        found / len(q2) if found >= 2 else 0.0,
        coverage,
        _bigram_jaccard(query_bigrams, text_bigrams),
    )


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
    """用近期保底与长期概率两路确定语义检索范围，再合并词面候选。"""

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
        recency_enabled: bool = True,
        recency_window_events: int = 80,
        recency_top_k: int = 40,
        full_scan_event_limit: int = 1000,
        long_term_probability_floor: float = 0.05,
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
        english_word_lexical: bool = False,
        event_index_level: str = "L1",
        detail_enabled: bool = True,
        detail_max_count: int = 1,
        detail_min_similarity: float | None = None,
        date_anchor_locale: str = "zh",
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
        self._full_scan_event_limit = max(0, int(full_scan_event_limit))
        self._long_term_probability_floor = min(
            max(float(long_term_probability_floor), 0.0), 0.99,
        )
        self._rng = random.Random()
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
        self._english_word_lexical = english_word_lexical
        if event_index_level not in {"L0", "L1"}:
            raise ValueError("event_index_level must be L0 or L1")
        self._event_index_level = event_index_level
        self._detail_enabled = detail_enabled
        self._detail_max_count = max(1, int(detail_max_count))
        self._detail_min_similarity = (
            self._semantic_min if detail_min_similarity is None else float(detail_min_similarity)
        )
        self._date_anchor_locale = date_anchor_locale if date_anchor_locale in {"zh", "en"} else "zh"

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
        candidate_sink: list[RecallCandidate] | None = None,
        candidate_call: int = 0,
    ) -> tuple[RecalledFragment, ...]:
        """交回事件摘要和对象事件事实；对象角色权重只参与排序。

        ``candidate_sink`` 非空时把本次按序访问到的候选追加进去（仅诊断用途）。
        """
        # 档位不改变细节条数，避免调用方自动升降把明细带多。锚点不参与检索。
        del level, anchor_event_ids
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
        small_library = bool(events) and len(events) <= self._full_scan_event_limit
        recent_event_ids: set[str] = set()
        semantic: dict[tuple[str, str, str], float] = {}
        if use_semantic:
            query_vector = self._embedding.embed_query(q)
            if small_library:
                # In a small library every event can participate. Sampling here
                # made old but relevant memories disappear between identical calls.
                semantic = self._semantic_signals(
                    subject_id, q, wanted, time_range,
                    query_vector=query_vector,
                    event_ids=tuple(event.event_id for event in events),
                    top_k=max(self._top_k, len(events) * 32),
                )
                if self._recency_enabled:
                    recent_event_ids = {
                        ev.event_id for ev in events[-self._recency_window_events:]
                    } if self._recency_window_events > 0 else set()
            elif self._recency_enabled:
                recent_events, long_term_events = self._split_semantic_pools(events, now)
                recent_event_ids = {ev.event_id for ev in recent_events}
                if recent_events:
                    semantic.update(self._semantic_signals(
                        subject_id, q, wanted, time_range,
                        query_vector=query_vector,
                        event_ids=tuple(ev.event_id for ev in recent_events),
                        top_k=max(1, self._recency_top_k),
                    ))
                if long_term_events:
                    long_term_signals = self._semantic_signals(
                        subject_id, q, wanted, time_range,
                        query_vector=query_vector,
                        event_ids=tuple(ev.event_id for ev in long_term_events),
                        top_k=self._top_k,
                    )
                    for key, score in long_term_signals.items():
                        semantic[key] = max(semantic.get(key, 0.0), score)
                logger.debug(
                    "recall semantic pools: eligible=%d recent=%d long_term_selected=%d",
                    len(events), len(recent_events), len(long_term_events),
                )
            elif events:
                semantic = self._semantic_signals(
                    subject_id, q, wanted, time_range, query_vector=query_vector,
                )
        candidates: list[_Candidate] = []
        for ev in events:
            candidates.extend(self._event_candidates(
                ev, q, wanted, semantic, time_range, now,
                interlocutor_object_id=interlocutor_object_id,
                use_semantic=use_semantic, use_lexical=use_lexical,
                recent_event_ids=recent_event_ids,
            ))
        candidates.extend(self._unclosed_candidates(
            subject_id, q, wanted, time_range,
            interlocutor_object_id=interlocutor_object_id,
            use_lexical=use_lexical,
        ))
        admitted = [c for c in candidates if self._admitted(c, use_semantic, use_lexical)]
        ordered = self._arrange(admitted, wanted, close=0.0 if small_library else None)
        if recent_event_ids and not small_library:
            ordered = self._reserve_recent_candidate(ordered, wanted)
        fragments = self._fill_budget(
            ordered, budget, expand_raw=expand_raw, limit=limit,
            candidate_sink=candidate_sink, candidate_call=candidate_call,
            query=q, query_vector=query_vector if use_semantic else None,
        )
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

    def backfill_index(self, subject_id: str | None = None) -> tuple[int, int]:
        """Reconcile active events with the current embedding collection.

        Returns (events scanned, points written). Safe to rerun: index_event
        compares source hashes, embedding version and object payload first.
        """
        scanned = 0
        written = 0
        for event in self._event_repo.list_all():
            if subject_id is not None and event.subject_id != subject_id:
                continue
            scanned += 1
            written += self.index_event(event)
        return scanned, written

    def index_event(self, event: Event) -> int:
        """L0 分块索引事件，或兼容旧 L1 事件与对象事实索引。"""
        if event.refinement_hidden_by:
            for key in self._vector_store.list_event_logical_ids(event.event_id):
                self._vector_store.delete(key)
            return 0
        object_ids = [role.role_id for role in event.role_list if not role.is_subject]
        object_ids.extend(event.refinement_objects.values())
        event_object_ids = list(dict.fromkeys((
            *object_ids,
            *(item.object_id for item in event.interlocutor_attributions if item.object_id),
        )))
        pending: list[tuple[str, str, dict]] = []
        desired: set[str] = set()
        l1 = retrieval_text(event, self._event_index_level)
        if l1:
            splitter = getattr(self._embedding, "split_document", None)
            chunks = splitter(l1) if splitter and self._event_index_level == "L0" else [l1]
            for index, chunk in enumerate(chunks):
                logical_id = event.event_id if index == 0 else f"{event.event_id}#raw#{index}"
                desired.add(logical_id)
                payload = self._index_payload(
                    event, material_type="event", object_id=None,
                    object_ids=event_object_ids, text=chunk,
                )
                payload["chunk_index"] = index
                if not self._index_unchanged(logical_id, payload):
                    pending.append((logical_id, chunk, payload))
        else:
            self._vector_store.delete(event.event_id)
        if self._role_repo is not None and self._event_index_level == "L1":
            for role in event.role_list:
                if role.is_subject:
                    continue
                logical_id = f"{event.event_id}#fact#{role.role_id}"
                fact = self._fact_l1(role.role_id, event.event_id)
                if not fact:
                    self._vector_store.delete(logical_id)
                    continue
                desired.add(logical_id)
                payload = self._index_payload(
                    event, material_type="object_event_fact", object_id=role.role_id,
                    object_ids=[role.role_id], text=fact,
                )
                if not self._index_unchanged(logical_id, payload):
                    pending.append((logical_id, fact, payload))
        if event.memory_stage and event.source_events:
            for leaf_id, leaf_text in self._source_raw_texts(event):
                logical_id = f"{event.event_id}#detail#{leaf_id}"
                desired.add(logical_id)
                payload = self._index_payload(
                    event, material_type="event_detail", object_id=None,
                    object_ids=event_object_ids, text=leaf_text,
                )
                payload["source_event_id"] = leaf_id
                payload["representation_level"] = "raw"
                if not self._index_unchanged(logical_id, payload):
                    pending.append((logical_id, leaf_text, payload))
        lister = getattr(self._vector_store, "list_event_logical_ids", None)
        if lister is not None:
            for stale_id in set(lister(event.event_id)) - desired:
                self._vector_store.delete(stale_id)
        if not pending:
            return 0
        vectors = self._embedding.embed_documents([text for _pid, text, _payload in pending])
        for (logical_id, _text, payload), vector in zip(pending, vectors):
            self._vector_store.upsert(logical_id, vector, payload)
        return len(pending)

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
        from ..services.refinement_service import utc
        return {
            "type": material_type,
            "subject_id": event.subject_id,
            "event_id": event.event_id,
            "object_id": object_id,
            "object_ids": list(object_ids),
            "representation_level": self._event_index_level if material_type == "event" else "L1",
            "source_hash": material_source_hash(text),
            "embedding_model": self._embedding_model,
            "embedding_version": self._embedding_version,
            "occurred_at": moment.timestamp() if moment else None,
            "event_time_start": utc(event.event_time_start or moment).timestamp(),
            "event_time_end": utc(event.event_time_end or moment).timestamp(),
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
            and existing.get("representation_level") == payload["representation_level"]
            and existing.get("object_id") == payload.get("object_id")
            and existing.get("object_ids") == payload.get("object_ids")
            and existing.get("event_time_start") == payload.get("event_time_start")
            and existing.get("event_time_end") == payload.get("event_time_end")
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
        if ev.event_time_start is not None and ev.event_time_end is not None:
            from ..services.refinement_service import utc
            return utc(ev.event_time_start) <= utc(end) and utc(ev.event_time_end) >= utc(start)
        moment = ev.occurred_at or ev.create_time
        if moment is None:
            return False
        return start <= moment <= end

    def _matches_objects(self, subject_id: str, ev: Event, wanted: tuple[str, ...]) -> bool:
        if not wanted:
            return True
        if any(oid in wanted for oid in ev.refinement_objects.values()):
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
        *,
        query_vector: list[float] | None = None,
        event_ids: tuple[str, ...] | None = None,
        top_k: int | None = None,
    ) -> dict[tuple[str, str, str], float]:
        if event_ids is not None and not event_ids:
            return {}
        vector = query_vector if query_vector is not None else self._embedding.embed_query(query)
        ranges = None
        if time_range is not None:
            start, end = time_range
            from ..services.refinement_service import utc
            ranges = {"event_time_start": (None, utc(end).timestamp()),
                      "event_time_end": (utc(start).timestamp(), None)}
        hits = self._vector_store.search(
            vector,
            top_k=self._top_k if top_k is None else top_k,
            payload_filter={
                "subject_id": subject_id,
                "embedding_model": self._embedding_model,
                "embedding_version": self._embedding_version,
            },
            any_of={
                **({"object_ids": list(wanted)} if wanted else {}),
                **({"event_id": list(event_ids)} if event_ids is not None else {}),
            } or None,
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
            if material == "event_detail":
                # 原文向量只给所属上层事件加分，第三段是子事件编号，不单独成为候选。
                object_id = str(payload.get("source_event_id") or "")
            elif material != "object_event_fact":
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
        recent_event_ids: set[str],
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
        raws = self._source_raw_texts(ev) if ev.memory_stage else []
        lexical_parts = [ev.content_raw or ""] if ev.memory_stage == 0 else [
            "\n".join(levels.values()), *[text for _leaf_id, text in raws],
        ]
        lexical = 0.0
        if use_lexical:
            lexical = max(
                (lexical_score(query, text, english_words=self._english_word_lexical)
                 for text in lexical_parts if text and text.strip()),
                default=0.0,
            )
        own_semantic = semantic.get(("event", ev.event_id, ""), 0.0) if use_semantic else 0.0
        detail_scores = {
            key[2]: score for key, score in semantic.items()
            if key[0] == "event_detail" and key[1] == ev.event_id
        } if use_semantic else None
        event_semantic = max(
            [own_semantic, *(detail_scores or {}).values()],
            default=0.0,
        )
        out = [_Candidate(
            material_type="event",
            event_id=ev.event_id,
            object_id=_shown_object_id(ev, wanted),
            semantic=event_semantic,
            lexical=lexical,
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
            is_recent_lane=ev.event_id in recent_event_ids,
            detail_scores=detail_scores,
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
                lexical=lexical_score(query, fact_text, english_words=self._english_word_lexical) if use_lexical else 0.0,
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
                is_recent_lane=ev.event_id in recent_event_ids,
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
                lexical=lexical_score(query, text, english_words=self._english_word_lexical),
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

    def _arrange(
        self, admitted: list[_Candidate], wanted: tuple[str, ...], *, close: float | None = None,
    ) -> list[_Candidate]:
        del wanted
        close = self._relevance_close if close is None else close
        return order_by_relevance(admitted, close)

    @staticmethod
    def _reserve_recent_candidate(
        ordered: list[_Candidate], wanted: tuple[str, ...],
    ) -> list[_Candidate]:
        """Reserve the first eligible recent result without changing vector scores."""
        del wanted
        if not ordered:
            return ordered
        recent_index = next(
            (idx for idx, item in enumerate(ordered) if item.is_recent_lane),
            None,
        )
        if recent_index is None or recent_index == 0:
            return ordered
        # Recency is a tie-breaker, not a licence to put an unrelated memory
        # ahead of a clearly better match.
        if ordered[recent_index].rank_score < 0.9 * ordered[0].rank_score:
            return ordered
        return [
            ordered[recent_index],
            *ordered[:recent_index],
            *ordered[recent_index + 1 :],
        ]

    def _fill_budget(
        self,
        ordered: list[_Candidate],
        budget: int,
        *,
        expand_raw: bool,
        limit: int | None,
        candidate_sink: list[RecallCandidate] | None = None,
        candidate_call: int = 0,
        query: str = "",
        query_vector: list[float] | None = None,
    ) -> list[RecalledFragment]:
        remaining = budget
        seen: set[tuple[str, str, str]] = set()
        out: list[RecalledFragment] = []

        def record(
            rank: int,
            cand: _Candidate,
            *,
            selected: bool,
            reason: str,
            level_name: str | None = None,
            chars: int = 0,
        ) -> None:
            """每条被访问到的候选都留痕，选中的与淘汰的都记（诊断用，不改结果）。"""
            if candidate_sink is None:
                return
            candidate_sink.append(RecallCandidate(
                call=candidate_call,
                rank=rank,
                material_type=cand.material_type,
                event_id=cand.event_id,
                object_id=cand.object_id,
                score=round(cand.rank_score, 6),
                selected=selected,
                reason=reason,
                representation_level=level_name,
                chars=chars,
                budget_remaining=remaining,
            ))

        for rank, cand in enumerate(ordered, 1):
            if limit is not None and len(out) >= limit:
                record(rank, cand, selected=False, reason="limit")
                break
            key = (cand.material_type, cand.event_id, cand.object_id or "")
            if key in seen:
                record(rank, cand, selected=False, reason="duplicate")
                continue
            seen.add(key)
            dates = sorted({value[:10] for value in (cand.event.source_dates if cand.event else [])})
            date_note = source_date_anchor(dates, self._date_anchor_locale)
            if cand.material_type == "unclosed":
                picked = (cand.raw, "raw") if len(cand.raw) <= remaining else None
            else:
                picked = select_representation(
                    cand.levels, cand.raw, remaining - len(date_note),
                    expand_raw=expand_raw or bool(cand.event and cand.event.memory_stage == 0),
                )
            if picked is None:
                record(rank, cand, selected=False, reason="no_room")
                continue
            body, level_name = picked
            detail_ids: list[str] = []
            detail_levels: list[str] = []
            detail_text = ""
            if cand.material_type == "event" and cand.event is not None and cand.event.memory_stage:
                detail_budget = remaining - len(date_note) - len(body) - 2
                detail_text, detail_ids, detail_levels = self._attach_details(
                    cand.event, query, detail_budget,
                    scores=cand.detail_scores,
                    rank=rank, parent=cand, candidate_sink=candidate_sink,
                    candidate_call=candidate_call,
                )
            head = date_note + body
            detail_content_start = (len(head) + 2) if detail_text else None
            content = head + (("\n\n" + detail_text) if detail_text else "")
            remaining -= len(content)
            record(rank, cand, selected=True, reason="selected",
                   level_name=level_name, chars=len(content))
            signals = {
                "semantic": round(cand.semantic, 6),
                "lexical": round(cand.lexical, 6),
                "object": cand.object_hit,
            }
            if cand.object_hit:
                signals["object_role_weight"] = cand.object_role_weight
            if cand.temporal is not None:
                signals["temporal"] = cand.temporal
            signals["recent_lane"] = 1.0 if cand.is_recent_lane else 0.0
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
                detail_event_ids=detail_ids,
                detail_levels=detail_levels,
                detail_content_start=detail_content_start,
                signals=signals,
                occurred_at=cand.occurred_at,
            ))
        return out

    def _attach_details(
        self,
        event: Event,
        query: str,
        budget: int,
        *,
        scores: dict[str, float] | None,
        rank: int,
        parent: _Candidate,
        candidate_sink: list[RecallCandidate] | None,
        candidate_call: int,
    ) -> tuple[str, list[str], list[str]]:
        """相似度达到门槛的第一层原文才附上。没有语义分时不现算向量。放不下的整条不附。"""
        if not self._detail_enabled or budget < 0 or scores is None:
            return "", [], []
        materials = [
            (child, level_name, text)
            for child, level_name, text in self._detail_materials(event)
            if level_name == "raw"
        ]
        if not materials:
            return "", [], []
        ranked = sorted(materials, key=lambda item: (
            scores.get(item[0].event_id, 0.0),
            lexical_score(query, item[2], english_words=self._english_word_lexical),
        ), reverse=True)
        chosen: list[tuple[Event, str, str]] = []
        room = budget
        for child, level_name, text in ranked:
            similarity = scores.get(child.event_id, 0.0)
            if similarity < self._detail_min_similarity:
                self._record_detail(candidate_sink, candidate_call, rank, parent, child,
                                    reason="detail_below_threshold", level_name=level_name,
                                    similarity=similarity)
                continue
            if len(chosen) >= self._detail_max_count:
                self._record_detail(candidate_sink, candidate_call, rank, parent, child,
                                    reason="detail_limit", level_name=level_name,
                                    similarity=similarity)
                continue
            block = self._detail_block(child, level_name, text)
            separator = 2 if chosen else 0
            if separator + len(block) > room:
                self._record_detail(candidate_sink, candidate_call, rank, parent, child,
                                    reason="detail_no_room", level_name=level_name,
                                    budget_remaining=room, similarity=similarity)
                continue
            room -= separator + len(block)
            chosen.append((child, level_name, block))
            self._record_detail(candidate_sink, candidate_call, rank, parent, child,
                                reason="detail_attached", level_name=level_name,
                                chars=len(block), budget_remaining=room, similarity=similarity)
        chosen.sort(key=lambda item: self._detail_order(item[0]))
        return ("\n\n".join(block for _child, _level, block in chosen),
                [child.event_id for child, _level, _block in chosen],
                [level for _child, level, _block in chosen])

    def _source_raw_texts(self, event: Event, seen: set[str] | None = None) -> list[tuple[str, str]]:
        """上层事件名下每一条第一层原文。中间层摘要不在这里返回。"""
        seen = set() if seen is None else seen
        found: list[tuple[str, str]] = []
        for event_id in event.source_events or []:
            if event_id in seen:
                continue
            seen.add(event_id)
            child = self._event_repo.get(event_id)
            if child is None:
                continue
            if child.memory_stage == 0 or not child.source_events:
                text = (child.content_raw or "").strip()
                if text:
                    found.append((child.event_id, text))
                continue
            found.extend(self._source_raw_texts(child, seen))
        return found

    def _detail_materials(self, event: Event, seen: set[str] | None = None) -> list[tuple[Event, str, str]]:
        seen = set() if seen is None else seen
        found: list[tuple[Event, str, str]] = []
        for event_id in event.source_events or []:
            if event_id in seen:
                continue
            seen.add(event_id)
            child = self._event_repo.get(event_id)
            if child is None:
                continue
            if child.memory_stage == 0:
                text = (child.content_raw or "").strip()
                if text:
                    found.append((child, "raw", text))
                continue
            summary = (child.summaries or {}).get("L1") or ""
            if summary.strip():
                found.append((child, "L1", summary.strip()))
            found.extend(self._detail_materials(child, seen))
        return found

    @staticmethod
    def _detail_order(event: Event) -> tuple[int, int, str]:
        if event.memory_sequence is not None:
            return event.memory_sequence, event.memory_sequence, event.event_id
        if event.memory_sequence_start is not None:
            return event.memory_sequence_start, event.memory_sequence_end or 0, event.event_id
        moment = event.event_time_start or event.occurred_at or event.create_time
        return 10**12, int(moment.timestamp()) if moment else 0, event.event_id

    def _detail_block(self, event: Event, level_name: str, text: str) -> str:
        kind = "原文" if level_name == "raw" else "摘要"
        if event.memory_sequence is not None:
            number = f"#{event.memory_sequence}"
        elif event.memory_sequence_start is not None and event.memory_sequence_end is not None:
            number = f"#{event.memory_sequence_start}–{event.memory_sequence_end}"
        else:
            number = ""
        dates = list(event.source_dates or [])
        if not dates and event.occurred_at is not None:
            dates = [event.occurred_at.date().isoformat()]
        anchor = source_date_anchor(dates, self._date_anchor_locale)
        label = f"{kind} {number}" if number else kind
        return f"{label}{anchor}{text}"

    def _record_detail(
        self,
        sink: list[RecallCandidate] | None,
        call: int,
        rank: int,
        parent: _Candidate,
        child: Event,
        *,
        reason: str,
        level_name: str,
        chars: int = 0,
        budget_remaining: int = 0,
        similarity: float | None = None,
    ) -> None:
        if sink is None:
            return
        sink.append(RecallCandidate(
            call=call, rank=rank, material_type="event", event_id=child.event_id,
            object_id=parent.object_id, score=round(parent.rank_score, 6),
            selected=reason == "detail_attached", reason=reason,
            representation_level=level_name, chars=chars, budget_remaining=budget_remaining,
            similarity=None if similarity is None else round(similarity, 6),
        ))

    def _fact_l1(self, role_id: str, event_id: str) -> str:
        levels = self._fact_levels(role_id, event_id)
        return levels.get("L1") or ""

    def _fact_levels(self, role_id: str, event_id: str) -> dict[str, str]:
        event = self._event_repo.get(event_id)
        if event and role_id in event.refinement_object_levels:
            return dict(event.refinement_object_levels[role_id])
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
            if (e.subject_id or "") == subject_id and not e.is_tombstoned and not e.refinement_hidden_by
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
        """近因通道：窗口内按 create_time 降序给 RRF 排名（docs/recall.md）。"""
        if not events:
            return {}
        window = max(1, self._recency_window_events)
        k = max(1, self._recency_top_k)
        ordered = sorted(events, key=lambda ev: ev.create_time, reverse=True)[:window]
        return {ev.event_id: idx + 1 for idx, ev in enumerate(ordered[:k])}

    def _split_semantic_pools(
        self, events: list[Event], now: datetime,
    ) -> tuple[list[Event], list[Event]]:
        """Keep the newest N eligible events searchable; sample older events by accessibility."""
        if not events:
            return [], []
        ordered = sorted(events, key=lambda ev: ev.create_time, reverse=True)
        recent_count = max(0, int(self._recency_window_events))
        recent = ordered[:recent_count]
        long_term = [
            ev for ev in ordered[recent_count:]
            if self._rng.random() < self._search_probability(self._effective_factor(ev, now))
        ]
        return recent, long_term

    def _search_probability(self, accessibility: float) -> float:
        """Map the unbounded accessibility factor into a bounded Bernoulli probability."""
        access = max(float(accessibility), 0.0)
        if math.isnan(access):
            access = 0.0
        normalized = 1.0 if math.isinf(access) else access / (1.0 + access)
        floor = self._long_term_probability_floor
        return floor + (1.0 - floor) * normalized

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
