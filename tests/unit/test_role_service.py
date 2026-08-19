"""Tests for RoleService (registration, white-painting, resolution)."""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.models.event import (
    BasicEmotionVector,
    EmotionalModel,
    Event,
    EventRoleEntry,
    Importance,
    RoleSnapshot,
)
from rems.services.role_service import RoleService
from rems.skills.role_extraction import RoleExtractionSkill
from rems.storage.database import Database
from rems.storage.repository import RoleRepository

from tests.conftest import FakeLLM


@pytest.fixture()
def role_service(config: REMSConfig, db: Database, fake_llm: FakeLLM):
    role_repo = RoleRepository(db)
    role_skill = RoleExtractionSkill(fake_llm, config)
    return RoleService(config, role_repo, role_skill)


class TestRegisterRole:
    def test_register_new(self, role_service: RoleService):
        role = role_service.register_role("Alice", entity_type="person")
        assert role.role_id.startswith("ROL-")
        assert role.name == "Alice"

    def test_register_idempotent(self, role_service: RoleService):
        r1 = role_service.register_role("Bob")
        r2 = role_service.register_role("Bob")
        assert r1.role_id == r2.role_id

    def test_find_role(self, role_service: RoleService):
        role_service.register_role("Carol")
        found = role_service.find_role("Carol")
        assert found is not None
        assert found.name == "Carol"

    def test_list_roles(self, role_service: RoleService):
        role_service.register_role("Dave")
        role_service.register_role("Eve")
        roles = role_service.list_roles()
        names = {r.name for r in roles}
        assert "Dave" in names
        assert "Eve" in names


class TestWhitePainting:
    def test_update_from_event(self, role_service: RoleService):
        role = role_service.register_role("Frank")

        event = Event(
            content_raw="Frank went to the market.",
            role_list=[
                EventRoleEntry(
                    role_id=role.role_id,
                    importance=Importance.A,
                    role_snapshot=RoleSnapshot(
                        l1_mention="Frank appeared",
                        l2_interaction="Frank went shopping",
                    ),
                    emotional_model=EmotionalModel.from_emotion(BasicEmotionVector(joy=0.5)),
                ),
            ],
        )

        role_service.update_from_event(event)

        summary = role_service.get_white_painting_summary(role.role_id)
        assert "Frank went shopping" in summary

        stored = role_service.get_role(role.role_id)
        assert stored is not None
        assert stored.white_painting[0].base_forgetting_factor == pytest.approx(25.0)

    def test_abstract_events_skipped(self, role_service: RoleService):
        role = role_service.register_role("Ghost")
        event = Event(
            content_raw="abstract content",
            is_abstract=True,
            role_list=[EventRoleEntry(role_id=role.role_id)],
        )
        role_service.update_from_event(event)
        summary = role_service.get_white_painting_summary(role.role_id)
        assert summary == ""
