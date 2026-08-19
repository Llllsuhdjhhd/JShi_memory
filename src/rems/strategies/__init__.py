"""Stable extension points for REMS algorithm choices.

Strategies keep long-lived service APIs stable while allowing prompt, recall,
abstraction, and forgetting algorithms to evolve behind small interfaces.
"""

from .abstraction import AbstractionEvidencePolicy, LeafContentRawEvidencePolicy
from .forgetting import (
    DefaultWhitePaintingRetentionStrategy,
    ForgettingScore,
    WhitePaintingRetentionStrategy,
)
from .recall import (
    DefaultRecallScoringStrategy,
    DefaultSummaryTierPolicy,
    RecallScoreBreakdown,
    RecallScoringStrategy,
    SummaryTierPolicy,
)

__all__ = [
    "AbstractionEvidencePolicy",
    "DefaultRecallScoringStrategy",
    "DefaultSummaryTierPolicy",
    "DefaultWhitePaintingRetentionStrategy",
    "ForgettingScore",
    "LeafContentRawEvidencePolicy",
    "RecallScoreBreakdown",
    "RecallScoringStrategy",
    "SummaryTierPolicy",
    "WhitePaintingRetentionStrategy",
]
