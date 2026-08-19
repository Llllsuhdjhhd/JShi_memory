from __future__ import annotations

from typing import Any, Protocol

from .types import RecallContext


class RecallBackend(Protocol):
    """Pluggable multi-channel recall strategy (RagProfile implementations)."""

    @property
    def profile_name(self) -> str:
        ...

    @property
    def rrf_channels(self) -> list[str]:
        """Channel names participating in RRF fusion for this profile."""

    def auxiliary_channel_ranks(
        self,
        ctx: RecallContext,
        *,
        tri_band_event_ids: list[str],
    ) -> dict[str, dict[str, int]]:
        """Extra channels beyond tri_band (e.g. bm25, recency). Maps channel -> {event_id: rank}."""

    def on_recall_trace(self) -> dict[str, Any] | None:
        """Optional diagnostics appended to recall_trace."""
