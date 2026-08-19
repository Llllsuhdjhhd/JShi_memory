"""Tests for EventService (seal, persist, index)."""

from __future__ import annotations


import pytest

from rems.config import REMSConfig
from rems.models.event import Event, EventRoleEntry, Importance
from rems.services.event_service import EventService
from rems.skills.event_enrichment import EventEnrichmentSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.storage.database import Database
from rems.storage.repository import EventRepository
from rems.embedding.tri_band import TriBandEncoder
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.vector_store import VectorStore

from tests.conftest import FakeLLM


def _make_vector_store(config: REMSConfig, db: Database) -> VectorStore:
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector_store = VectorStore(config)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    vector_store.set_tri_band(tri_band)
    return vector_store


@pytest.fixture()
def event_service(config: REMSConfig, db: Database, fake_llm: FakeLLM, tmp_dir):
    event_repo = EventRepository(db)
    vector_store = _make_vector_store(config, db)
    role_skill = RoleExtractionSkill(fake_llm, config)
    enrichment_skill = EventEnrichmentSkill(fake_llm, config, role_fallback=role_skill)
    return EventService(config, fake_llm, event_repo, vector_store, enrichment_skill)
    


class TestSealEvent:
    def test_seal_creates_event(self, event_service: EventService, fake_llm: FakeLLM):
        fake_llm.push_response({"summaries": {"L1": "核心事实"}})
        fake_llm.push_response("装饰文本")

        event = event_service.seal_event(
            "今天天气晴朗，适合散步。",
            skip_roles=True,
        )

        assert isinstance(event, Event)
        assert event.event_id.startswith("EVT-")
        assert event.content_raw == "今天天气晴朗，适合散步。"
        assert event.status.value == "active"

    def test_seal_persists(self, event_service: EventService, fake_llm: FakeLLM):
        fake_llm.push_response({"summaries": {"L1": "fact"}})
        fake_llm.push_response("decor")

        event = event_service.seal_event("persisted text", skip_roles=True)
        loaded = event_service.get_event(event.event_id)

        assert loaded is not None
        assert loaded.content_raw == "persisted text"

    def test_seal_with_roles(self, event_service: EventService, fake_llm: FakeLLM):
        fake_llm.push_response({"summaries": {"L1": "s"}})
        fake_llm.push_response("d")

        roles = [EventRoleEntry(role_id="ROL-test", importance=Importance.A)]
        event = event_service.seal_event("some event", role_entries=roles)
        assert len(event.role_list) == 1
        assert event.role_list[0].role_id == "ROL-test"
