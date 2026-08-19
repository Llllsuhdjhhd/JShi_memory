from __future__ import annotations

from typing import Any

from ...config import REMSConfig
from ...storage.repository import EventRepository
from ..bm25_index import EventBm25Index
from ..channels import bm25_channel_ranks, recency_channel_ranks
from ..protocol import RecallBackend
from ..types import RecallContext


class HybridLiteraryBackend:
    """Long-form literary narrative: tri-band + BM25 + recency (hybrid light)."""

    profile_name = "hybrid_literary"

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        bm25_index: EventBm25Index,
    ) -> None:
        self._config = config
        self._event_repo = event_repo
        self._bm25 = bm25_index
        self._rrf_channels = list(config.recall_rrf_channels) or [
            "tri_band",
            "bm25",
            "recency",
        ]

    @property
    def rrf_channels(self) -> list[str]:
        return self._rrf_channels

    def auxiliary_channel_ranks(
        self,
        ctx: RecallContext,
        *,
        tri_band_event_ids: list[str],
    ) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        bm25 = bm25_channel_ranks(ctx, self._bm25, self._config)
        if bm25:
            out["bm25"] = bm25
        rec = recency_channel_ranks(
            self._event_repo,
            self._config,
            active_event_ids=ctx.active_event_ids,
        )
        if rec:
            out["recency"] = rec
        return out

    def on_recall_trace(self) -> dict[str, Any] | None:
        return {
            "profile": self.profile_name,
            "channels": self.rrf_channels,
            "bm25_docs": len(self._bm25._event_ids),  # noqa: SLF001 — trace only
        }
