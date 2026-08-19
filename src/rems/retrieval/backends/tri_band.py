from __future__ import annotations

from typing import Any

from ...config import REMSConfig
from ..protocol import RecallBackend
from ..types import RecallContext


class TriBandOnlyBackend:
    """Default profile: vector tri-band only (behavior-equivalent to pre-refactor path)."""

    profile_name = "tri_band"
    rrf_channels = ["tri_band"]

    def __init__(self, config: REMSConfig) -> None:
        self._config = config

    def auxiliary_channel_ranks(
        self,
        ctx: RecallContext,
        *,
        tri_band_event_ids: list[str],
    ) -> dict[str, dict[str, int]]:
        return {}

    def on_recall_trace(self) -> dict[str, Any] | None:
        return {"profile": self.profile_name, "channels": self.rrf_channels}
