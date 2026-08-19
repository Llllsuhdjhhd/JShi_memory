from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from ..models.event import Event, Importance
from ..strategies.forgetting import DefaultWhitePaintingRetentionStrategy

if TYPE_CHECKING:
    from ..config import REMSConfig
    from ..storage.repository import EventRepository, RoleRepository


_IMPORTANCE_RANK = {
    Importance.S: 5,
    Importance.A: 4,
    Importance.B: 3,
    Importance.C: 2,
    Importance.D: 1,
}


def _best_importance(entries: list) -> Importance | None:
    if not entries:
        return None
    return max((e.importance for e in entries), key=lambda imp: _IMPORTANCE_RANK.get(imp, 0))


class EventWeightDeriver:
    """Derive event-level w_i from white-painting effective_forgetting (§1.1.5)."""

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        role_repo: RoleRepository,
    ):
        self._config = config
        self._event_repo = event_repo
        self._role_repo = role_repo
        self._forgetting = DefaultWhitePaintingRetentionStrategy(config)

    def derive_wi(self, event: Event, *, now: datetime | None = None) -> float:
        if getattr(event, "ptsd_immune", False):
            return float(self._config.recall_forgetting_factor_cap)

        if event.is_abstract:
            leaf_ids = self._event_repo.resolve_basic_event_ids(event.event_id)
            best = 0.0
            for eid in leaf_ids:
                leaf = self._event_repo.get(eid)
                if leaf:
                    best = max(best, self.derive_wi(leaf, now=now))
            return best

        if not event.role_list:
            return 0.0

        top_imp = _best_importance(event.role_list)
        if top_imp is None:
            return 0.0

        candidates = [
            e for e in event.role_list
            if _IMPORTANCE_RANK.get(e.importance, 0) == _IMPORTANCE_RANK.get(top_imp, 0)
        ]
        best_w = 0.0
        for entry in candidates:
            wp = self._role_repo.get_white_painting_by_event(entry.role_id, event.event_id)
            if wp is None:
                continue
            score = self._forgetting.score(wp, is_penalized=False, now=now)
            best_w = max(best_w, score.effective_forgetting)
        return best_w

    def derive_w_eff(self, event: Event, asf_i: float, *, now: datetime | None = None) -> float:
        w_i = self.derive_wi(event, now=now)
        asf = min(max(float(asf_i), 0.0), 0.999999)
        return max(w_i * (1.0 - asf), 1e-6)


def event_id_to_point_id(event_id: str) -> str:
    """Stable UUID for Qdrant point id from REMS event_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, event_id))
