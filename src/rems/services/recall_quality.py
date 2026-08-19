from __future__ import annotations

import logging

from ..config import REMSConfig

logger = logging.getLogger(__name__)


class RecallQualityController:
    """Rolling EMA over abstraction coherence (**a**) and recall-block relevance (**b**).

    ``combined_signal`` feeds ``semantic_distance_cap()`` when dynamic distance is enabled.
    Weight ``recall_quality_weight_coherence_vs_relevance`` applies to **a**; default is small so **b** dominates once observed.

    ``ema_a`` / ``ema_b`` 仅驻留在 **Pipeline 进程内存**：未写入 SQLite/Chroma；
    ``REMSPipeline`` 重启或新建后即丢失，下一轮冷启动再回到 ``neutral``。
    """

    def __init__(self, config: REMSConfig):
        self._config = config
        self._ema_a: float | None = None
        self._ema_b: float | None = None

    def reset(self) -> None:
        self._ema_a = None
        self._ema_b = None

    def record_abstraction_coherence_rate(self, rate_a: float) -> None:
        cfg = self._config
        r = max(0.0, min(1.0, float(rate_a)))
        alpha = max(1e-6, min(1.0, cfg.recall_quality_ema_alpha))
        if self._ema_a is None:
            self._ema_a = r
        else:
            self._ema_a = alpha * r + (1.0 - alpha) * self._ema_a
        logger.debug(
            "recall-quality EMA coherence rate_a=%.3f → ema_a=%.3f",
            rate_a,
            self._ema_a,
        )

    def record_recall_block_relevance_rate(self, rate_b: float) -> None:
        cfg = self._config
        r = max(0.0, min(1.0, float(rate_b)))
        alpha = max(1e-6, min(1.0, cfg.recall_quality_ema_alpha))
        if self._ema_b is None:
            self._ema_b = r
        else:
            self._ema_b = alpha * r + (1.0 - alpha) * self._ema_b
        logger.debug(
            "recall-quality EMA relevance rate_b=%.3f → ema_b=%.3f",
            rate_b,
            self._ema_b,
        )

    def combined_signal(self) -> float:
        cfg = self._config
        has_a = self._ema_a is not None
        has_b = self._ema_b is not None
        if has_a and has_b:
            w = max(0.0, min(1.0, cfg.recall_quality_weight_coherence_vs_relevance))
            return w * self._ema_a + (1.0 - w) * self._ema_b
        if has_a:
            return self._ema_a
        if has_b:
            return self._ema_b
        return cfg.recall_dynamic_distance_quality_neutral

    def semantic_distance_cap(self) -> float | None:
        if not self._config.recall_dynamic_distance_enabled:
            return None

        cfg = self._config
        combined = self.combined_signal()
        neutral = cfg.recall_dynamic_distance_quality_neutral
        base = cfg.recall_dynamic_distance_cap_base
        sens = cfg.recall_dynamic_distance_sensitivity

        adjusted = base + sens * (combined - neutral)
        floor_v = cfg.recall_dynamic_distance_cap_floor
        ceiling_v = cfg.recall_dynamic_distance_cap_ceiling
        if floor_v > ceiling_v:
            floor_v, ceiling_v = ceiling_v, floor_v
        cap = max(floor_v, min(ceiling_v, adjusted))
        logger.debug(
            "recall-quality semantic cap combined=%.3f → cap=%.4f",
            combined,
            cap,
        )
        return cap

    def should_audit_relevance(self, ingest_cycle: int) -> bool:
        n = self._config.recall_relevance_audit_every_n_ingests
        if n <= 0 or ingest_cycle <= 0:
            return False
        return ingest_cycle % n == 0
