from __future__ import annotations

from typing import Any

from ...config import REMSConfig
from ...storage.repository import EventRepository, RoleRepository
from ..bm25_index import EventBm25Index
from ..channels import bm25_channel_ranks, recency_channel_ranks
from ..rrf import ranks_from_scores
from ..types import RecallContext


class NpcAgentBackend:
    """NPC dialogue: tri-band + role-name BM25 hint + light recency."""

    profile_name = "npc_agent"

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        bm25_index: EventBm25Index,
    ) -> None:
        self._config = config
        self._event_repo = event_repo
        self._role_repo = role_repo
        self._bm25 = bm25_index
        self._rrf_channels = list(config.recall_rrf_channels) or [
            "tri_band",
            "role_anchor",
            "recency",
        ]

    @property
    def rrf_channels(self) -> list[str]:
        return self._rrf_channels

    def _role_anchor_ranks(self, ctx: RecallContext) -> dict[str, int]:
        if not ctx.focus_role_ids:
            return {}
        scores: dict[str, float] = {}
        for ev in self._event_repo.list_all(exclude_tombstoned=True):
            if ev.status.value == "silent":
                continue
            overlap = sum(
                1 for entry in ev.role_list if entry.role_id in ctx.focus_role_ids
            )
            if overlap > 0:
                scores[ev.event_id] = float(overlap)
        return ranks_from_scores(scores, reverse=True)

    def auxiliary_channel_ranks(
        self,
        ctx: RecallContext,
        *,
        tri_band_event_ids: list[str],
    ) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        role_r = self._role_anchor_ranks(ctx)
        if role_r:
            out["role_anchor"] = role_r
        rec = recency_channel_ranks(
            self._event_repo,
            self._config,
            active_event_ids=ctx.active_event_ids,
        )
        if rec and self._config.recall_recency_enabled:
            out["recency"] = rec
        if self._config.recall_bm25_enabled:
            bm25 = bm25_channel_ranks(ctx, self._bm25, self._config)
            if bm25:
                out["bm25"] = bm25
        return out

    def on_recall_trace(self) -> dict[str, Any] | None:
        return {"profile": self.profile_name, "channels": self.rrf_channels}
