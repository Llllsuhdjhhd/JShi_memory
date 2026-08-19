from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..config import REMSConfig
from ..models.event import Event


@dataclass(frozen=True)
class RecallScoreBreakdown:
    """Observable components of a recall score."""

    cosine: float
    time_decay: float
    role_boost: float
    arousal: float
    activation_energy: float
    total: float


class RecallScoringStrategy(Protocol):
    """Score a candidate event after vector search."""

    def score(self, event: Event, cosine_sim: float) -> RecallScoreBreakdown:
        """Return score components and final total."""
        ...


class SummaryTierPolicy(Protocol):
    """Choose summary-detail offset for recall rendering."""

    def tier_offset(self, event: Event, focus_role_ids: set[str]) -> int:
        """Return offset relative to the event's middle summary tier."""
        ...


class DefaultRecallScoringStrategy:
    """Current hybrid scoring policy extracted behind an interface."""

    def __init__(self, config: REMSConfig):
        self._config = config

    def score(self, event: Event, cosine_sim: float) -> RecallScoreBreakdown:
        time_decay = self.time_decay(event.create_time)
        role_boost = self.role_boost(event)
        arousal = event.affective_energy
        act = event.activation_energy
        arousal_w = 0.10
        act_w = 0.05
        cosine_w = max(0.0, 1.0 - arousal_w - act_w - 0.15 - 0.20)
        total = (
            cosine_w * cosine_sim
            + 0.15 * time_decay
            + 0.20 * role_boost
            + arousal_w * arousal
            + act_w * act
        )
        return RecallScoreBreakdown(
            cosine=cosine_sim,
            time_decay=time_decay,
            role_boost=role_boost,
            arousal=arousal,
            activation_energy=act,
            total=total,
        )

    @staticmethod
    def time_decay(create_time: datetime, half_life_days: float = 30.0) -> float:
        age_seconds = (datetime.now() - create_time).total_seconds()
        age_days = max(age_seconds / 86400, 0.0)
        return math.exp(-0.693 * age_days / half_life_days)

    @staticmethod
    def role_boost(event: Event) -> float:
        boost = 0.0
        importance_weights = {"S": 0.3, "A": 0.2, "B": 0.1, "C": 0.05, "D": 0.0}
        for entry in event.role_list:
            key = entry.importance.value if hasattr(entry.importance, "value") else str(entry.importance)
            boost = max(boost, importance_weights.get(key, 0.0))
        return boost


class DefaultSummaryTierPolicy:
    """Current role-aware tier policy extracted behind an interface."""

    def __init__(self, config: REMSConfig):
        self._config = config

    def tier_offset(self, event: Event, focus_role_ids: set[str]) -> int:
        cfg = self._config
        if not focus_role_ids:
            return cfg.recall_default_tier_offset

        best = "absent"
        rank = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "absent": 5}
        for entry in event.role_list:
            if entry.role_id not in focus_role_ids:
                continue
            imp = entry.importance.value if hasattr(entry.importance, "value") else str(entry.importance)
            if rank.get(imp, 5) < rank.get(best, 5):
                best = imp

        if best in ("S", "A"):
            return cfg.recall_default_tier_offset - cfg.recall_primary_role_detail_shift
        if best == "B":
            return cfg.recall_default_tier_offset
        return cfg.recall_default_tier_offset + cfg.recall_minor_role_compress_shift
