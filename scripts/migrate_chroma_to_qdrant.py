#!/usr/bin/env python3
"""Migrate legacy Chroma-indexed events to Qdrant tri-band vectors."""

from __future__ import annotations

import argparse

from rems.config import REMSConfig
from rems.embedding.tri_band import TriBandEncoder
from rems.jobs.tier1_refresh import refresh_tier1
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.tier1_store import Tier1Store
from rems.storage.active_pool_cache import ActivePoolCache
from rems.storage.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate events to Qdrant + Tier-1")
    parser.add_argument("--config-env", default=None, help="Optional .env path (unused placeholder)")
    args = parser.parse_args()
    _ = args

    config = REMSConfig()
    if config.embedding.provider != "hash":
        config.embedding.provider = "hash"

    db = Database(config.storage.database_url)
    db.create_tables()
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    tier1 = Tier1Store(db, config)
    cache = ActivePoolCache(tier1, config)
    vector = VectorStore(config)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    vector.set_tri_band(tri_band)

    count = 0
    for event in event_repo.list_all(exclude_tombstoned=True):
        vector.upsert_event_vectors(event)
        count += 1

    refresh_tier1(config, tier1, event_repo, role_repo, cache)
    print(f"Migrated {count} events to Qdrant tri-band + Tier-1")


if __name__ == "__main__":
    main()
