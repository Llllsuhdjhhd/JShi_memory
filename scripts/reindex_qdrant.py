#!/usr/bin/env python3
"""Rebuild recall vectors from SQLite events (no re-ingest).

用真实（或 hash）嵌入把全部非墓碑事件重新索引进 Qdrant 本地向量库（design/1010）。
换嵌入模型 / 改检索文本规则 / 向量维度变化后，跑本脚本重建即可。

Example::

    python scripts/reindex_qdrant.py \\
        --db rems.db \\
        --qdrant-path qdrant_data \\
        --embedding local \\
        --model BAAI/bge-base-zh-v1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rems.config import REMSConfig, StorageConfig
from rems.recall import (
    HashEmbedding,
    QdrantRecallVectorStore,
    RecallPipeline,
    SentenceTransformerEmbedding,
    versioned_collection_name,
)
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reindex all non-tombstoned events into Qdrant (recall, design/1010)"
    )
    parser.add_argument("--db", required=True, help="SQLite database path")
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
    parser.add_argument(
        "--model",
        default=None,
        help="Override embedding model name (default: REMSConfig embedding.model_name)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: REMSConfig storage.qdrant_collection)",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    storage_kwargs: dict = {"database_url": f"sqlite:///{db_path.as_posix()}"}
    if args.qdrant_path:
        storage_kwargs["qdrant_path"] = str(Path(args.qdrant_path).resolve())
    if args.collection:
        storage_kwargs["qdrant_collection"] = args.collection

    config = REMSConfig(storage=StorageConfig(**storage_kwargs))
    if args.embedding:
        config.embedding.provider = args.embedding
    if args.model:
        config.embedding.model_name = args.model

    embedding_model = (
        "hash" if config.embedding.provider == "hash" else config.embedding.model_name
    )
    embedding = (
        HashEmbedding()
        if config.embedding.provider == "hash"
        else SentenceTransformerEmbedding(config.embedding.model_name)
    )
    vector_store = QdrantRecallVectorStore(
        versioned_collection_name(
            config.storage.qdrant_collection,
            config.recall_embedding_version,
        ),
        url=config.storage.qdrant_url,
        path=config.storage.qdrant_path,
    )

    db = Database(config.storage.database_url)
    event_repo = EventRepository(db)
    pipe = RecallPipeline(
        embedding,
        vector_store,
        event_repo,
        role_repo=RoleRepository(db),
        embedding_model=embedding_model,
        embedding_version=config.recall_embedding_version,
    )

    events = event_repo.list_all(exclude_tombstoned=True)
    for event in events:
        pipe.index_event(event)

    print(
        f"Reindexed {len(events)} events → Qdrant collection '{vector_store._collection}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
