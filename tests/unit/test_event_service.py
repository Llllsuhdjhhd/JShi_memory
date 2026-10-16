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

from tests.conftest import FakeLLM


@pytest.fixture()
def event_service(config: REMSConfig, db: Database, fake_llm: FakeLLM, tmp_dir):
    event_repo = EventRepository(db)
    role_skill = RoleExtractionSkill(fake_llm, config)
    enrichment_skill = EventEnrichmentSkill(fake_llm, config, role_fallback=role_skill)
    return EventService(config, fake_llm, event_repo, None, enrichment_skill)
    

def test_l1_budget_is_independent_of_auxiliary_allocations(event_service):
    # L1 must keep its one-third allowance even when auxiliary budgets change.
    baseline = event_service._compute_budget(2000, role_count_estimate=1)
    assert baseline.summary_level_budgets["L1"] == 666
    cfg = event_service._config
    cfg.budget_ratio_summary = 0.05
    cfg.compression_target_ratio = 0.05
    cfg.compression_budget_multiplier = 0.2
    changed = event_service._compute_budget(2000, role_count_estimate=20)
    assert changed.summary_level_budgets == baseline.summary_level_budgets
    assert changed.snapshot_level_budgets != baseline.snapshot_level_budgets


def test_l1_ratio_can_be_tuned_without_expanding_short_original(event_service):
    event_service._config.summary_l1_retention_ratio = 0.5
    assert event_service._compute_budget(2000).summary_level_budgets["L1"] == 1000
    assert event_service._compute_budget(10).summary_level_budgets["L1"] == 10


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
