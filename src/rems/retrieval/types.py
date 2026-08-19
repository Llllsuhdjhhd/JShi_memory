from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models.event import EventRoleEntry
from ..models.metabolism import Shadow


@dataclass(frozen=True)
class RecallContext:
    """Per-ingest retrieval context passed to RecallBackend implementations."""

    query: str
    shadow: Shadow | None
    act_source: str
    search_text: str
    focus_role_ids: set[str]
    focus_role_entries: list[EventRoleEntry]
    active_event_ids: list[str] | None = None
    ingest_seq: int | None = None


@dataclass
class RecallCandidate:
    """Unified candidate after multi-channel fusion (before factor/mood modifiers)."""

    event_id: str
    channel_ranks: dict[str, int] = field(default_factory=dict)
    tri_band_distance: float = 1.0
    effective_weight: float = 0.0
    raw_scores: dict[str, Any] = field(default_factory=dict)
