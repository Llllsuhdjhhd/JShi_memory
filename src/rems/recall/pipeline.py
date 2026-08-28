"""回忆检索管线（design/1010 §3）：多路召回 + RRF + 遗忘/情感调制 + 可选精排。"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from ..models.event import Event
from ..port import RecalledFragment
from ..storage.repository import EventRepository, ObjectTimelineRepository
from ..utils.zh_normalize import normalize_for_substring_match as _norm
from .intent import RecallIntent, RuleIntentClassifier

logger = logging.getLogger(__name__)


_EMOTION_LEXICON = {
    "positive": ["开心", "高兴", "快乐", "喜悦", "喜欢", "满意", "幸福", "开心", "愉快"],
    "negative": ["难过", "伤心", "悲伤", "痛苦", "害怕", "恐惧", "生气", "愤怒", "焦虑", "失望"],
}


def retrieval_text(event: Event) -> str:
    """事件检索文本：L1 摘要（或原文）+ 对象 id + 地点（design/1010 §2.4）。"""
    parts = [event.summaries.get("L1") or event.content_raw]
    names = [r.role_id for r in event.role_list if not r.is_subject]
    if names:
        parts.append("对象:" + " ".join(names))
    if event.location:
        parts.append("地点:" + event.location)
    return "\n".join(parts)


class RecallPipeline:
    """查询理解 → 多路召回（语义 / 词法 / 对象 / 时间 / 锚点）→ RRF + 调制 → 精排 → 档位选择。"""

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
        channels: tuple[str, ...] | None = None,
        reinforce: bool = True,
    ) -> tuple[RecalledFragment, ...]:
        """实现 MemoryBackendPort.recall（只读 + 记忆恢复）。"""
        q = _norm((query or "").strip())
        if not q:
            return ()

        # 查询理解（规则版）：对象 / 情绪 / 近因 / 事实意图
        intent: RecallIntent | None = None
        if self._intent_classifier is not None:
            intent = self._intent_classifier.classify(q)
            if not object_id and intent.object_id:
                object_id = intent.object_id

        events = self._candidate_events(subject_id)
        if not events:
            return ()

        # 多路召回：语义 / 词法 / 对象 / 时间 / 锚点
        semantic_ids = self._semantic_route(subject_id, q, object_id=object_id)
        lexical_scores = self._lexical_scores(q, events)
        object_ids = self._object_route(subject_id, object_id, events)
        recency_scores = self._recency_route(events) if self._recency_enabled else {}
        anchor_ids = set(anchor_event_ids or ())
        for eid in anchor_event_ids:
            ev = self._event_repo.get(eid)
            if ev is not None:
                anchor_ids.update(ev.split_prefix_event_ids or [])

        active = set(channels) if channels else {
            "semantic", "lexical", "object", "recency", "anchor",
        }
        now = datetime.now()
        scores: dict[str, float] = {}
        for ev in events:
            eff = self._effective_factor(ev, now)
            if eff < self._silence_threshold:
                continue
            rrf = 0.0
            if "semantic" in active and ev.event_id in semantic_ids:
                rrf += 1.0 / (self._rrf_k + semantic_ids[ev.event_id])
            if "lexical" in active and ev.event_id in lexical_scores:
                rrf += 1.0 / (self._rrf_k + lexical_scores[ev.event_id])
            if "object" in active and ev.event_id in object_ids:
                rrf += 1.0 / (self._rrf_k + object_ids[ev.event_id])
            if "recency" in active and ev.event_id in recency_scores:
                rrf += 1.0 / (self._rrf_k + recency_scores[ev.event_id])
            if "anchor" in active and ev.event_id in anchor_ids:
                rrf += 1.0 / (self._rrf_k + 1)
            if rrf <= 0:
                continue
            score = rrf * (eff ** self._factor_alpha)
            score *= self._emotion_modifier(q, ev, intent=intent)
            scores[ev.event_id] = score

        if not scores:
            return ()
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]

        # 精排（可选）
        if self._reranker is not None and ranked:
            cand_events = [self._event_repo.get(eid) for eid, _ in ranked]
            cand_texts = [self._retrieval_text(e) for e in cand_events if e is not None]
            rerank_scores = self._reranker.rerank(q, cand_texts)
            if rerank_scores:
                ordered = sorted(
                    zip(ranked, rerank_scores),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                ranked = [item for item, _ in ordered]

        fragments: list[RecalledFragment] = []
        for eid, sc in ranked:
            ev = self._event_repo.get(eid)
            if ev is None:
                continue
            content, summary_level = self._select_summary(ev, level)
            fragments.append(RecalledFragment(
                event_id=ev.event_id,
                text=ev.content_raw,
                content=content,
                kind=ev.origin,
                object_id=object_id,
                source_ids=list(ev.source_ids or []),
                score=round(sc, 6),
                summary_level=summary_level,
            ))
            # 记忆恢复：命中事件强化遗忘因子（唯一写操作；recall_test 可关）
            if reinforce:
                self._reinforce(ev)
            if limit is not None and len(fragments) >= limit:
                break
        return tuple(fragments)

    # ------------------------------------------------------------------
    def index_event(self, event: Event) -> None:
        """封存时写向量（design/1010 §4：写入时索引）。"""
        text = self._retrieval_text(event)
        vector = self._embedding.embed_documents([text])[0]
        self._vector_store.upsert(
            event.event_id,
            vector,
            payload={
                "subject_id": event.subject_id,
                "object_ids": [r.role_id for r in event.role_list if not r.is_subject],
                "create_time": event.create_time.timestamp(),
            },
        )

    # ------------------------------------------------------------------
    def _candidate_events(self, subject_id: str) -> list[Event]:
        return [
            e for e in self._event_repo.list_all(exclude_tombstoned=True)
            if (e.subject_id or "") == subject_id and not e.is_tombstoned
        ]

    def _retrieval_text(self, ev: Event) -> str:
        return retrieval_text(ev)

    def _semantic_route(
        self, subject_id: str, query: str, *, object_id: str | None
    ) -> dict[str, int]:
        qv = self._embedding.embed_query(query)
        payload_filter: dict = {"subject_id": subject_id}
        if object_id:
            payload_filter["object_ids"] = object_id
        hits = self._vector_store.search(qv, top_k=self._top_k, payload_filter=payload_filter)
        return {h["event_id"]: idx + 1 for idx, h in enumerate(hits) if h.get("event_id")}

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        t = _norm(text)
        return {t[i : i + 2] for i in range(max(0, len(t) - 1))}

    def _lexical_scores(self, query: str, events: list[Event]) -> dict[str, int]:
        qb = self._bigrams(query)
        if not qb:
            return {}
        scored = [
            (
                ev.event_id,
                self._bigram_jaccard(
                    qb,
                    self._bigrams(ev.content_raw + " " + (ev.summaries.get("L1") or "")),
                ),
            )
            for ev in events
        ]
        scored = [(eid, n) for eid, n in scored if n > 0]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return {eid: idx + 1 for idx, (eid, _) in enumerate(scored)}

    @staticmethod
    def _bigram_jaccard(a: set[str], b: set[str]) -> float:
        """Bigram Jaccard：命中数 / 并集数，压制长摘要/长原文的天然优势。"""
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

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
        return float(getattr(ev, "forgetting_factor", 1.0) or 1.0) * decay

    def _emotion_modifier(
        self, query: str, ev: Event, intent: RecallIntent | None = None,
    ) -> float:
        if ev.emotion is None:
            return 1.0
        if intent is not None and intent.emotion:
            want_pos = intent.emotion == "positive"
            want_neg = intent.emotion == "negative"
        else:
            q = _norm(query)
            want_pos = any(w in q for w in _EMOTION_LEXICON["positive"])
            want_neg = any(w in q for w in _EMOTION_LEXICON["negative"])
        if not want_pos and not want_neg:
            return 1.0
        ev_pos = ev.emotion.valence > 0
        match = (want_pos and ev_pos) or (want_neg and not ev_pos)
        return 1.0 + self._mood_beta if match else 1.0 - self._mood_beta

    def _select_summary(self, ev: Event, level: int) -> tuple[str, str | None]:
        key = f"L{max(1, level)}"
        if key not in (ev.summaries or {}):
            key = ev.mid_summary_key if ev.summaries else "L1"
        text = (ev.summaries or {}).get(key) or ev.content_raw
        return text, key

    def _reinforce(self, ev: Event) -> None:
        factor = min(
            float(getattr(ev, "forgetting_factor", 1.0) or 1.0) * self._reinforce_multiplier,
            self._reinforce_cap,
        )
        self._event_repo.update_forgetting_factor(ev.event_id, factor)
