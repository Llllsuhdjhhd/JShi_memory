from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import REMSConfig
    from .tier1_store import Tier1Store


class ActivePoolCache:
    """In-process cache of top sample_key event IDs for O(1) RAG pre-filter."""

    def __init__(self, tier1: Tier1Store, config: REMSConfig):
        self._tier1 = tier1
        self._config = config
        self._lock = threading.Lock()
        self._ids: list[str] = []

    def refresh(self, limit: int | None = None) -> list[str]:
        base = limit or self._config.tier1.active_pool_base
        ids = self._tier1.fetch_active_ids(base)
        with self._lock:
            self._ids = ids
        return ids

    def fetch(self, limit: int | None = None) -> list[str]:
        with self._lock:
            cached = list(self._ids)
        if not cached:
            return self.refresh(limit)
        if limit is not None and len(cached) > limit:
            return cached[:limit]
        return cached

    def replace(self, ids: list[str]) -> None:
        with self._lock:
            self._ids = list(ids)
