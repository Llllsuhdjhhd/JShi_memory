from __future__ import annotations

"""Rolling-window latency monitor with dual-channel load factors (§4.6)."""

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class PhaseStat:
    name: str
    tolerance_ms: float
    window_size: int = 32
    samples: deque[float] = field(default_factory=deque)

    def record(self, elapsed_ms: float) -> None:
        if self.window_size <= 0:
            return
        self.samples.append(elapsed_ms)
        while len(self.samples) > self.window_size:
            self.samples.popleft()

    @property
    def mean_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    @property
    def overload_ratio(self) -> float:
        if self.tolerance_ms <= 0 or not self.samples:
            return 0.0
        return self.mean_ms / self.tolerance_ms


class PerfMonitor:
    """Dual-channel performance monitor: recall vs abstract (§4.6)."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_size: int = 32,
        tolerances_ms: dict[str, float] | None = None,
        load_factor_max: float = 4.0,
        recall_phases: list[str] | None = None,
        abstract_phases: list[str] | None = None,
        silence_boost_at_max: float = 4.0,
    ) -> None:
        self.enabled = enabled
        self._window_size = window_size
        self._load_factor_max = max(1.0, float(load_factor_max))
        self._silence_boost_at_max = max(1.0, float(silence_boost_at_max))
        self._recall_phases = set(recall_phases or ["rag_search", "recall_assembly"])
        self._abstract_phases = set(abstract_phases or ["abstraction_mining", "narrative_dedupe"])
        self._stats: dict[str, PhaseStat] = {}
        for name, tol in (tolerances_ms or {}).items():
            self._stats[name] = PhaseStat(name=name, tolerance_ms=float(tol), window_size=window_size)
        self._default_tolerance_ms = 200.0

    def record(self, phase: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        stat = self._stats.get(phase)
        if stat is None:
            stat = PhaseStat(
                name=phase,
                tolerance_ms=self._default_tolerance_ms,
                window_size=self._window_size,
            )
            self._stats[phase] = stat
        stat.record(elapsed_ms)

    @contextmanager
    def timer(self, phase: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(phase, (time.perf_counter() - start) * 1000.0)

    def stats_snapshot(self) -> dict[str, PhaseStat]:
        return dict(self._stats)

    def mean_ms(self, phase: str) -> float:
        stat = self._stats.get(phase)
        return stat.mean_ms if stat else 0.0

    def _load_factor_for_phases(self, phases: set[str]) -> float:
        if not self.enabled or not self._stats:
            return 1.0
        ratios = [
            s.overload_ratio for s in self._stats.values()
            if s.name in phases and s.samples
        ]
        if not ratios:
            return 1.0
        max_ratio = max(ratios)
        if max_ratio <= 1.0:
            return 1.0
        return min(self._load_factor_max, max_ratio)

    def load_factor(self) -> float:
        """Legacy global max across all phases."""
        if not self.enabled or not self._stats:
            return 1.0
        max_ratio = max((s.overload_ratio for s in self._stats.values() if s.samples), default=0.0)
        if max_ratio <= 1.0:
            return 1.0
        return min(self._load_factor_max, max_ratio)

    def load_factor_recall(self) -> float:
        return self._load_factor_for_phases(self._recall_phases)

    def load_factor_abstract(self) -> float:
        return self._load_factor_for_phases(self._abstract_phases)

    def is_overloaded(self) -> bool:
        return self.load_factor() > 1.0

    def adjusted_active_pool_limit(self, base: int, min_limit: int) -> int:
        lf = self.load_factor_recall()
        floor_k = max(1, int(min_limit))
        if lf <= 1.0 or self._load_factor_max <= 1.0:
            return max(floor_k, int(base))
        t = (lf - 1.0) / (self._load_factor_max - 1.0)
        k = int(base - (base - floor_k) * t)
        return max(floor_k, k)

    def adjusted_recent_k(self, default_k: int, min_k: int, *, channel: str = "abstract") -> int:
        lf = self.load_factor_abstract() if channel == "abstract" else self.load_factor_recall()
        floor_k = max(1, int(min_k))
        if lf <= 1.0 or self._load_factor_max <= 1.0:
            return max(floor_k, int(default_k))
        t = (lf - 1.0) / (self._load_factor_max - 1.0)
        k = int(default_k - (default_k - floor_k) * t)
        return max(floor_k, k)

    def adjusted_silence_threshold(self, base_threshold: float) -> float:
        """Deprecated: 2026.06 uses Tier-1 pool instead. Returns base unchanged."""
        return base_threshold
