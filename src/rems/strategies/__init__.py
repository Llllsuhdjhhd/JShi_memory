"""Stable extension points for REMS algorithm choices.

Strategies keep long-lived service APIs stable while allowing prompt, recall,
abstraction, and forgetting algorithms to evolve behind small interfaces.
"""

from .forgetting import (
    DefaultWhitePaintingRetentionStrategy,
    ForgettingScore,
    WhitePaintingRetentionStrategy,
)

__all__ = [
    "DefaultWhitePaintingRetentionStrategy",
    "ForgettingScore",
    "WhitePaintingRetentionStrategy",
]
