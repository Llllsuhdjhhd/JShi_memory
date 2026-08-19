from __future__ import annotations

import logging
import math
from datetime import datetime

# Emotion & Adaptation (EMA) 动态演化（白皮书 2.5）。
# 情感基准由 role_extraction 的 8 维基础情绪产出；EMAEvolver 负责合成滚动
# valence / arousal / energy，并把事件级 activation_energy 写入 Event。

from ..config import REMSConfig
from ..models.event import Event, EventRoleEntry
from ..storage.repository import RoleRepository

logger = logging.getLogger(__name__)


class EMAEvolver:
    """Evolve per-role emotion state via time-decayed rolling affect.

    对每个角色读取近期白描，按 ``exp(-lambda * delta_hours)`` 计算滚动
    ``rolling_valence`` / ``rolling_arousal`` / ``energy``。
    """

    def __init__(self, config: REMSConfig, role_repo: RoleRepository):
        self._config = config
        self._repo = role_repo

    # ------------------------------------------------------------------
    def evolve_event(self, event: Event) -> None:
        """Mutate ``event`` in place: adapt each role's emotion + set activation energy."""
        if event.is_abstract:
            return

        for entry in event.role_list:
            self._evolve_role_entry(entry)

        if event.role_list:
            energy = max(entry.emotional_model.energy for entry in event.role_list)
            event.activation_energy = min(energy * self._config.activation_energy_gain, 1.0)

    # ------------------------------------------------------------------
    def _evolve_role_entry(self, entry: EventRoleEntry) -> None:
        try:
            recent = self._repo.get_white_painting(
                entry.role_id, limit=self._config.ema_history_window
            )
        except Exception:
            logger.debug("EMA: no history for %s", entry.role_id, exc_info=True)
            recent = []

        if not recent:
            entry.emotional_model.rolling_arousal = entry.emotional_model.arousal
            entry.emotional_model.rolling_valence = entry.emotional_model.valence
            entry.emotional_model.energy = entry.emotional_model.arousal
            return

        now = datetime.now()
        weighted_valence = entry.emotional_model.valence
        weighted_arousal = entry.emotional_model.arousal
        weight_sum = 1.0
        energy = entry.emotional_model.arousal
        decay_lambda = self._config.emotion_decay_lambda_per_hour

        for wp in recent:
            hours = max((now - wp.create_time).total_seconds() / 3600.0, 0.0)
            weight = math.exp(-decay_lambda * hours)
            weighted_valence += wp.emotional_model.valence * weight
            weighted_arousal += wp.emotional_model.arousal * weight
            energy += wp.emotional_model.arousal * weight
            weight_sum += weight

        entry.emotional_model.rolling_valence = weighted_valence / weight_sum
        entry.emotional_model.rolling_arousal = weighted_arousal / weight_sum
        entry.emotional_model.energy = energy
