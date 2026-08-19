"""Tests for RecallService (tri-band search, scoring, assembly)."""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.embedding.query_act_processor import QueryActResult
from rems.embedding.tri_band import TriBandEncoder
from rems.models.event import Event
from rems.models.metabolism import Shadow
from rems.services.recall_service import RecallService
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.vector_store import VectorStore


@pytest.fixture()
def recall_service(config: REMSConfig, db: Database):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector_store = VectorStore(config)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    vector_store.set_tri_band(tri_band)
    svc = RecallService(
        config, event_repo, role_repo, vector_store, tri_band=tri_band,
    )
    return svc, event_repo, vector_store


class TestRecallBlock:
    def test_empty_store_returns_empty(self, recall_service):
        svc, _, _ = recall_service
        block = svc.build_recall_block("hello")
        assert block.items == []
        assert block.total_length == 0

    def test_recall_finds_indexed_event(self, recall_service):
        svc, event_repo, vector_store = recall_service

        event = Event(content_raw="张三去了北京参加会议")
        event_repo.save(event)
        vector_store.upsert_event_vectors(event)

        block = svc.build_recall_block("北京的会议")
        assert len(block.items) >= 1
        assert block.items[0].event_id == event.event_id

    def test_context_package(self, recall_service):
        svc, _, _ = recall_service
        shadow = Shadow(content="previous context")
        pkg = svc.build_context_package("new input", shadow)
        assert pkg.current_input == "new input"
        assert pkg.shadow.content == "previous context"


class TestScoring:
    def test_time_decay(self):
        from datetime import datetime
        score = RecallService._time_decay(datetime.now())
        assert 0.99 < score <= 1.0

    def test_time_decay_old(self):
        from datetime import datetime, timedelta
        old = datetime.now() - timedelta(days=60)
        score = RecallService._time_decay(old, half_life_days=30)
        assert score < 0.3

    def test_single_stream_score_orders_by_distance(self, recall_service):
        svc, _, _ = recall_service
        a = Event(content_raw="A")
        b = Event(content_raw="B")
        stream = {
            a.event_id: (a, 0.2, 0.5),
            b.event_id: (b, 0.8, 0.5),
        }
        ranked = svc._single_stream_score(stream, [])
        assert ranked[0][0].event_id == a.event_id


class TestEmoBandSkip:
    def test_search_skips_emo_vector_band(self, recall_service, monkeypatch):
        svc, event_repo, vector_store = recall_service
        assert svc._config.tri_band.emo_search_enabled is False

        event = Event(content_raw="indexed event for emo skip test")
        event_repo.save(event)
        vector_store.upsert_event_vectors(event)

        calls: list[str] = []
        original = vector_store.qdrant._client.query_points

        def _track_query_points(*args, **kwargs):
            using = kwargs.get("using")
            if using:
                calls.append(using)
            return original(*args, **kwargs)

        monkeypatch.setattr(vector_store.qdrant._client, "query_points", _track_query_points)
        svc.build_recall_block("indexed event")

        assert "vector_act" in calls
        assert "vector_ent" in calls
        assert "vector_emo" not in calls


class TestQueryActProcessor:
    def test_query_act_audit_populated(self, recall_service):
        svc, _, _ = recall_service

        class _StubProcessor:
            def resolve_act_text(self, text: str) -> QueryActResult:
                return QueryActResult(act_text=f"ACT:{len(text)}", source="llm_compress")

        svc._query_act_processor = _StubProcessor()
        svc.build_recall_block("x" * 300)
        assert svc.query_act_audit is not None
        assert len(svc.query_act_audit["entries"]) >= 1
        assert svc.query_act_audit["entries"][0]["source"] == "llm_compress"

    def test_query_act_cache_deduplicates(self, recall_service):
        svc, _, _ = recall_service
        calls = {"n": 0}

        class _CountingProcessor:
            def resolve_act_text(self, text: str) -> QueryActResult:
                calls["n"] += 1
                return QueryActResult(act_text="cached", source="passthrough")

        svc._query_act_processor = _CountingProcessor()
        svc._query_act_source("same")
        svc._query_act_source("same")
        assert calls["n"] == 1

    def test_multi_route_uses_single_act_compress(self, recall_service, monkeypatch):
        svc, event_repo, _ = recall_service
        svc._config.recall_multi_route_enabled = True
        svc._config.recall_query_compress_enabled = True

        event = Event(content_raw="多路检索测试事件")
        event_repo.save(event)

        calls: list[str] = []

        class _TrackingProcessor:
            def resolve_act_text(self, text: str) -> QueryActResult:
                calls.append(text)
                return QueryActResult(act_text="UNIFIED_ACT", source="llm_compress")

        svc._query_act_processor = _TrackingProcessor()
        shadow = Shadow(content="残影铺垫内容足够长" * 20)
        svc.build_recall_block("当前块查询文本", shadow=shadow, focus_role_ids=set())

        assert len(calls) == 1
        assert "残影铺垫" in calls[0]
        assert svc.query_act_audit is not None
        assert len(svc.query_act_audit["entries"]) == 1

    def test_precomputed_act_source_skips_processor(self, recall_service):
        svc, _, _ = recall_service
        calls = {"n": 0}

        class _CountingProcessor:
            def resolve_act_text(self, text: str) -> QueryActResult:
                calls["n"] += 1
                return QueryActResult(act_text="should-not-run", source="llm_compress")

        svc._query_act_processor = _CountingProcessor()
        svc.build_recall_block(
            "query",
            shadow=Shadow(content="shadow bit"),
            act_source="PRECOMPUTED",
            act_source_label="llm_compress",
        )
        assert calls["n"] == 0
        assert svc.query_act_audit["entries"][0]["source"] == "llm_compress"
        assert svc.query_act_audit["entries"][0]["act_text"].startswith("PRECOMPUTED")
