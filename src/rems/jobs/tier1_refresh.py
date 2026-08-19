from __future__ import annotations

import logging

from ..config import REMSConfig
from ..storage.active_pool_cache import ActivePoolCache
from ..storage.repository import EventRepository, RoleRepository
from ..storage.tier1_store import Tier1Store
from ..strategies.event_weight import EventWeightDeriver

logger = logging.getLogger(__name__)


def refresh_tier1(
    config: REMSConfig,
    tier1: Tier1Store,
    event_repo: EventRepository,
    role_repo: RoleRepository,
    cache: ActivePoolCache | None = None,
) -> int:
    """Recompute all sample_key values and refresh active pool cache."""
    deriver = EventWeightDeriver(config, event_repo, role_repo)
    count = 0
    for event in event_repo.list_all(exclude_tombstoned=True):
        asf = tier1.get_asf(event.event_id)
        w_i = deriver.derive_wi(event)
        tier1.upsert_tier1(event.event_id, w_i=w_i, asf_i=asf)
        count += 1
    if cache is not None:
        cache.refresh()
    logger.info("Tier-1 refresh completed for %d events", count)
    return count


def main() -> None:
    from ..storage.database import Database

    config = REMSConfig()
    db = Database(config.storage.database_url)
    db.create_tables()
    tier1 = Tier1Store(db, config)
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    cache = ActivePoolCache(tier1, config)
    refresh_tier1(config, tier1, event_repo, role_repo, cache)


if __name__ == "__main__":
    main()
