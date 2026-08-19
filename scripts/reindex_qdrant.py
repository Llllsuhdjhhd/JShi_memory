#!/usr/bin/env python3
"""Rebuild Qdrant tri-band vectors from SQLite events (no re-ingest).

Use after changing embedding provider, act/ent text rules, or vector dims.
Existing SQLite events are re-encoded and upserted into Qdrant.

Example (scenario workspace)::

    python scripts/reindex_qdrant.py \\
        --db tests/scenarios/hongloumeng/outputs/continuous_run/rems_sim.db \\
        --qdrant-path tests/scenarios/hongloumeng/outputs/continuous_run/qdrant_sim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rems.config import REMSConfig, StorageConfig
from rems.embedding.tri_band import TriBandEncoder
from rems.models.event import EventStatus
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Reindex all ACTIVE events into Qdrant")
    parser.add_argument(
        "--db",
        required=True,
        help="SQLite database path (e.g. scenario rems_sim.db)",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Qdrant local path (default: REMSConfig storage.qdrant_path)",
    )
    parser.add_argument(
        "--embedding",
        choices=("local", "hash"),
        default=None,
        help="Override embedding provider (default: REMSConfig, usually local)",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    storage_kwargs: dict = {"database_url": f"sqlite:///{db_path.as_posix()}"}
    if args.qdrant_path:
        storage_kwargs["qdrant_path"] = str(Path(args.qdrant_path).resolve())

    config = REMSConfig(storage=StorageConfig(**storage_kwargs))
    if args.embedding:
        config.embedding.provider = args.embedding

    db = Database(config.storage.database_url)
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector = VectorStore(config)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    vector.set_tri_band(tri_band)

    events = event_repo.list_all(status=EventStatus.ACTIVE, exclude_tombstoned=True)
    count = 0
    for event in events:
        vector.upsert_event_vectors(event)
        count += 1

    print(f"Reindexed {count} ACTIVE events → Qdrant ({vector.count()} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
