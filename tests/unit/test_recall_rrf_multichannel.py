"""Multi-channel RRF fusion and BM25 index (RagProfile support)."""

from __future__ import annotations

from datetime import datetime, timedelta

from rems.config import REMSConfig, StorageConfig
from rems.models.event import Event, EventStatus
from rems.retrieval.bm25_index import EventBm25Index
from rems.retrieval.channels import bm25_channel_ranks, recency_channel_ranks, tri_band_ranks_from_hits
from rems.retrieval.rrf import reciprocal_rank_fusion
from rems.retrieval.types import RecallContext
from rems.storage.database import Database
from rems.storage.repository import EventRepository


def test_reciprocal_rank_fusion_merges_channels():
    ranks = {
        "tri_band": {"e1": 1, "e2": 5},
        "bm25": {"e2": 1, "e3": 2},
    }
    scores = reciprocal_rank_fusion(ranks, channels=["tri_band", "bm25"], k=60)
    assert scores["e1"] > scores["e3"]
    assert scores["e2"] > 0


def test_bm25_finds_keyword_overlap(tmp_path):
    cfg = REMSConfig(storage=StorageConfig(database_url=f"sqlite:///{tmp_path / 'bm25.db'}"))
    db = Database(cfg.storage.database_url)
    db.create_tables()
    repo = EventRepository(db)

    ev_old = Event(
        event_id="EVT-old",
        content_raw="冷香丸配方记在册上",
        summaries={"L1": "冷香丸配方"},
        create_time=datetime.now() - timedelta(days=10),
    )
    ev_new = Event(
        event_id="EVT-new",
        content_raw="今日又提到冷香丸",
        summaries={"L1": "宝玉问起冷香丸"},
        create_time=datetime.now(),
    )
    repo.save(ev_old)
    repo.save(ev_new)

    index = EventBm25Index()
    index.rebuild([ev_old, ev_new])
    ctx = RecallContext(
        query="冷香丸",
        shadow=None,
        act_source="冷香丸",
        search_text="冷香丸",
        focus_role_ids=set(),
        focus_role_entries=[],
    )
    cfg.recall_bm25_enabled = True
    cfg.recall_bm25_top_k = 10
    ranks = bm25_channel_ranks(ctx, index, cfg)
    assert ranks.get("EVT-old", 99) <= 2
    assert ranks.get("EVT-new", 99) <= 2


def test_recency_ranks_newer_first(tmp_path):
    cfg = REMSConfig(
        storage=StorageConfig(database_url=f"sqlite:///{tmp_path / 'rec.db'}"),
        recall_recency_enabled=True,
        recall_recency_window_events=10,
        recall_recency_top_k=5,
    )
    db = Database(cfg.storage.database_url)
    db.create_tables()
    repo = EventRepository(db)
    older = Event(event_id="EVT-a", content_raw="a", create_time=datetime.now() - timedelta(hours=2))
    newer = Event(event_id="EVT-b", content_raw="b", create_time=datetime.now())
    repo.save(older)
    repo.save(newer)
    ranks = recency_channel_ranks(repo, cfg)
    assert ranks["EVT-b"] < ranks["EVT-a"]


def test_tri_band_ranks_from_hits():
    hits = [
        {"event_id": "e2", "distance": 0.2},
        {"event_id": "e1", "distance": 0.1},
    ]
    ranks = tri_band_ranks_from_hits(hits)
    assert ranks["e1"] == 1
    assert ranks["e2"] == 2
