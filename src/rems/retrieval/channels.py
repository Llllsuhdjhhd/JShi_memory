from __future__ import annotations

from typing import Any

from ..config import REMSConfig
from ..models.event import Event
from ..storage.repository import EventRepository
from .bm25_index import EventBm25Index
from .rrf import ranks_from_scores
from .types import RecallContext


def tri_band_ranks_from_hits(hits: list[dict[str, Any]]) -> dict[str, int]:
    """1-based ranks from vector hits (lower distance = better rank)."""
    ordered = sorted(
        [h for h in hits if h.get("event_id")],
        key=lambda h: float(h.get("distance", 1.0)),
    )
    return {h["event_id"]: idx + 1 for idx, h in enumerate(ordered)}


def bm25_channel_ranks(
    ctx: RecallContext,
    index: EventBm25Index,
    config: REMSConfig,
) -> dict[str, int]:
    if not config.recall_bm25_enabled:
        return {}
    query = (ctx.act_source or ctx.search_text or ctx.query).strip()
    if not query:
        return {}
    eligible = set(ctx.active_event_ids) if ctx.active_event_ids else None
    top_k = config.recall_bm25_top_k
    hits = index.search(query, top_k=top_k, eligible_ids=eligible)
    scores = {eid: sc for eid, sc in hits}
    return ranks_from_scores(scores, reverse=True)


def recency_channel_ranks(
    event_repo: EventRepository,
    config: REMSConfig,
    *,
    active_event_ids: list[str] | None = None,
) -> dict[str, int]:
    if not config.recall_recency_enabled:
        return {}
    window = config.recall_recency_window_events
    top_k = config.recall_recency_top_k
    events = event_repo.list_all(exclude_tombstoned=True)
    eligible: set[str] | None = set(active_event_ids) if active_event_ids else None

    def _ok(ev: Event) -> bool:
        if ev.is_tombstoned or ev.status.value == "silent":
            return False
        if eligible is not None and ev.event_id not in eligible:
            return False
        return True

    recent = [e for e in events if _ok(e)]
    recent.sort(key=lambda e: e.create_time, reverse=True)
    recent = recent[:window]
    ranks: dict[str, int] = {}
    for idx, ev in enumerate(recent[:top_k]):
        ranks[ev.event_id] = idx + 1
    return ranks
