"""Tests for MetabolismService (shadow, unclosed events, triggers)."""

from __future__ import annotations


import pytest

from rems.config import REMSConfig
from rems.services.event_service import EventService
from rems.services.metabolism_service import MetabolismService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.event_enrichment import EventEnrichmentSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.storage.database import Database
from rems.storage.repository import EventRepository, MetabolismRepository

from ..conftest import FakeLLM


@pytest.fixture()
def metabolism_service(config: REMSConfig, db: Database, fake_llm: FakeLLM, vector_store):
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    role_skill = RoleExtractionSkill(fake_llm, config)
    enrichment_skill = EventEnrichmentSkill(fake_llm, config, role_fallback=role_skill)
    event_service = EventService(config, fake_llm, event_repo, vector_store, enrichment_skill)
    boundary_skill = BoundaryDetectionSkill(fake_llm, config)
    return MetabolismService(config, meta_repo, boundary_skill, event_service)


class TestProcessInput:
    def test_completed_event_sealed(self, metabolism_service: MetabolismService, fake_llm: FakeLLM):
        # boundary detection returns one completed event
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        # enrichment (summary + roles)
        fake_llm.push_response({
            "summaries": {"L1": "thing happened"},
            "roles": []
        })
        # decoration
        fake_llm.push_response("warm colours")

        events = metabolism_service.process_input("A thing happened.")
        assert len(events) == 1
        assert events[0].content_raw == "A thing happened."

    def test_no_completed_returns_empty(self, metabolism_service: MetabolismService, fake_llm: FakeLLM):
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed_indices": [1],
        })

        events = metabolism_service.process_input("partial text")
        assert events == []

    def test_force_save(self, metabolism_service: MetabolismService, fake_llm: FakeLLM):
        # enrichment
        fake_llm.push_response({
            "summaries": {"L1": "forced"},
            "roles": []
        })
        # decoration
        fake_llm.push_response("decor")

        events = metabolism_service.process_input("forced content", force_save=True)
        assert len(events) >= 1
