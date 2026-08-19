from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from ..config import REMSConfig
from ..models.role import WhitePaintingEntry

if TYPE_CHECKING:
    from ..observability import PerfMonitor


@dataclass(frozen=True)
class ForgettingScore:
    """Observable retention score for a white-painting entry."""

    is_penalized: bool
    retention: float
    score: float
    effective_forgetting: float
    is_silenced: bool


class WhitePaintingRetentionStrategy(Protocol):
    """Score white-painting entries for capacity-aware forgetting."""

    def score(
        self,
        entry: WhitePaintingEntry,
        *,
        is_penalized: bool,
        now: datetime | None = None,
    ) -> ForgettingScore:
        """Return a retention score for one timeline entry."""
        ...


class DefaultWhitePaintingRetentionStrategy:
    """Current FIFO + time half-life + AE policy as a swappable strategy.

    可选注入 ``perf_monitor``：当系统主算法（RAG / 回忆组装 / 抽象挖掘 / 叙事判重）
    平均耗时越线时，``adjusted_silence_threshold`` 会按 ``load_factor`` 抬高静默阈值——
    更多旧条目跌破阈值进入 SILENT，等价于"系统主动遗忘加速"（白皮书 §2.3 高负载自我保护）。
    没传 monitor 时退化为静态 ``forgetting_silence_threshold``。
    """

    def __init__(self, config: REMSConfig, perf_monitor: "PerfMonitor | None" = None):
        self._config = config
        self._perf = perf_monitor

    def _silence_threshold(self) -> float:
        base = self._config.forgetting_silence_threshold
        if self._perf is None:
            return base
        return self._perf.adjusted_silence_threshold(base)

    def score(
        self,
        entry: WhitePaintingEntry,
        *,
        is_penalized: bool,
        now: datetime | None = None,
    ) -> ForgettingScore:
        now = now or datetime.now()

        # 计算动态遗忘因子
        age_since_access = max((now - getattr(entry, "last_accessed_time", entry.create_time)).total_seconds() / 86400, 0.0)
        half_life = self._config.wp_half_life_days
        forgetting_decay = math.exp(-0.693 * age_since_access / half_life)
        effective_forgetting = float(getattr(entry, "forgetting_factor", 1.0)) * forgetting_decay
        is_silenced = effective_forgetting < self._silence_threshold()

        if not is_penalized:
            return ForgettingScore(
                is_penalized=False, 
                retention=1.0, 
                score=1.0, 
                effective_forgetting=effective_forgetting, 
                is_silenced=is_silenced
            )

        age_days = max((now - entry.create_time).total_seconds() / 86400, 0.0)
        retention = math.exp(-0.693 * age_days / half_life)
        score = entry.memory_weight * 0.4 + retention * 0.6
        return ForgettingScore(
            is_penalized=True, 
            retention=retention, 
            score=score * effective_forgetting,  # 动态遗忘因子影响最终评分
            effective_forgetting=effective_forgetting,
            is_silenced=is_silenced
        )
