from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..models.event import Event
from ..storage.tier1_store import Tier1Store
from ..strategies.event_weight import EventWeightDeriver

if TYPE_CHECKING:
    from ..config import REMSConfig
    from ..storage.repository import EventRepository, RoleRepository
    from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AbstractionShadowingService:
    """ASF propagation after abstract event synthesis (§3.4)."""

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        vector_store: VectorStore,
        tier1: Tier1Store,
    ):
        self._config = config
        self._event_repo = event_repo
        self._role_repo = role_repo
        self._vector = vector_store
        self._tier1 = tier1
        self._weight = EventWeightDeriver(config, event_repo, role_repo)

    def compute_single_hop_asf(
        self,
        leaf: Event,
        abstract: Event,
    ) -> float:
        cos = self._vector.cosine_between_events(leaf.event_id, abstract.event_id)
        len_a = max(len(leaf.content_raw), 1)
        len_b = max(len(abstract.content_raw), 1)
        ratio = min(1.0, len_b / len_a)
        return max(0.0, cos * ratio)

    def propagate(
        self,
        abstract_event: Event,
        shadow_lambda: float,
    ) -> None:
        lam = min(max(float(shadow_lambda), 0.0), 1.0)
        leaf_ids = self._event_repo.resolve_basic_event_ids(abstract_event.event_id)
        for leaf_id in leaf_ids:
            leaf = self._event_repo.get(leaf_id)
            if leaf is None or leaf.is_abstract:
                continue
            if getattr(leaf, "ptsd_immune", False):
                continue
            hop = self.compute_single_hop_asf(leaf, abstract_event)
            incoming = hop * lam
            old_asf = self._tier1.get_asf(leaf_id)
            survival_old = 1.0 - old_asf
            survival_new = survival_old * (1.0 - incoming)
            new_asf = min(0.999999, 1.0 - survival_new)
            w_i = self._weight.derive_wi(leaf)
            self._tier1.upsert_tier1(leaf_id, w_i=w_i, asf_i=new_asf)
        logger.debug(
            "ASF propagated from abstract %s to %d leaves (lambda=%.2f)",
            abstract_event.event_id,
            len(leaf_ids),
            lam,
        )
