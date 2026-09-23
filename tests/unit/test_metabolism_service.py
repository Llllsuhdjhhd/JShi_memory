"""Tests for MetabolismService (shadow, unclosed events, triggers)."""

from __future__ import annotations


from datetime import datetime, timedelta

import pytest

from rems.config import REMSConfig
from rems.models.metabolism import UnclosedEvent
from rems.port import MemoryExperience
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
        assert all(e.seal_reason == "truncated" for e in events)


class TestAbandonedUnclosedSeal:
    def test_hard_idle_days_seals_without_semantic_close(
        self, metabolism_service: MetabolismService, fake_llm: FakeLLM,
    ):
        past = datetime.now() - timedelta(days=8)
        metabolism_service._repo.save_unclosed_event(UnclosedEvent(
            id="UC-idle",
            content_fragments=["我们商量周末出门，还没定哪天"],
            created_at=past,
            updated_at=past,
            last_hit_time=past,
        ))
        fake_llm.push_response({
            "events": [],
            "residual": [{"indices": [1], "continues": "UC-idle"}],
            "no_form": [],
        })
        fake_llm.push_response({"summaries": {"L1": "商量出门未定"}, "roles": []})
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({"summaries": {"L1": "吃了面"}, "roles": []})

        memory = MemoryExperience(
            subject_id="jshi-1",
            text="今天吃了面",
            objects={},
            source_ids=("s1",),
            occurred_at=datetime.now(),
        )
        events = metabolism_service.process_input("今天吃了面", memory=memory)
        assert [e.content_raw for e in events] == [
            "我们商量周末出门，还没定哪天",
            "今天吃了面",
        ]
        assert events[0].seal_reason == "truncated"
        assert events[1].seal_reason == "closed"
        assert metabolism_service._repo.get_unclosed_events() == []

    def test_partial_idle_with_half_limit_length_seals(
        self, metabolism_service: MetabolismService, config: REMSConfig, fake_llm: FakeLLM,
    ):
        config.unclosed_char_limit = 100
        past = datetime.now() - timedelta(days=4)
        fragment = "甲" * 60  # > 0.5 * 100，但未达硬上限
        metabolism_service._repo.save_unclosed_event(UnclosedEvent(
            id="UC-partial",
            content_fragments=[fragment],
            created_at=past,
            updated_at=past,
            last_hit_time=past,
        ))
        fake_llm.push_response({
            "events": [],
            "residual": [{"indices": [1], "continues": "UC-partial"}],
            "no_form": [],
        })
        fake_llm.push_response({"summaries": {"L1": "半限闲置封存"}, "roles": []})
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({"summaries": {"L1": "新输入"}, "roles": []})

        memory = MemoryExperience(
            subject_id="jshi-1",
            text="今天继续",
            objects={},
            source_ids=("s1",),
            occurred_at=datetime.now(),
        )
        events = metabolism_service.process_input("今天继续", memory=memory)
        assert events[0].content_raw == fragment
        assert events[0].seal_reason == "truncated"
        assert metabolism_service._repo.get_unclosed_events() == []

    def test_recent_over_char_limit_stays_for_boundary(
        self, metabolism_service: MetabolismService, config: REMSConfig, fake_llm: FakeLLM,
    ):
        config.unclosed_char_limit = 20
        now = datetime.now()
        fragment = "x" * 20
        metabolism_service._repo.save_unclosed_event(UnclosedEvent(
            id="UC-long",
            content_fragments=[fragment],
            created_at=now,
            updated_at=now,
            last_hit_time=now,
        ))
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed_indices": [1, 2],
        })

        memory = MemoryExperience(
            subject_id="jshi-1",
            text="新的一句",
            objects={},
            source_ids=("s1",),
            occurred_at=now,
        )
        events = metabolism_service.process_input("新的一句", memory=memory)
        assert events == []
        leftover = metabolism_service._repo.get_unclosed_events()
        assert len(leftover) == 1
        assert fragment in leftover[0].merged_content

    def test_interlocutor_change_does_not_seal(
        self, metabolism_service: MetabolismService, fake_llm: FakeLLM,
    ):
        now = datetime.now()
        metabolism_service._repo.save_unclosed_event(UnclosedEvent(
            id="UC-a",
            content_fragments=["甲还在说周末计划"],
            interlocutor="OBJ-A",
            created_at=now,
            updated_at=now,
            last_hit_time=now,
        ))
        # 边界把旧残影 + 新输入都留在未完成；说话人已换也不应先封存旧条。
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed_indices": [1, 2],
        })

        memory = MemoryExperience(
            subject_id="jshi-1",
            text="乙说你好",
            objects={"乙": "OBJ-B"},
            interlocutor="OBJ-B",
            source_ids=("s1",),
            occurred_at=now,
        )
        events = metabolism_service.process_input("乙说你好", memory=memory)
        assert events == []
        leftover = metabolism_service._repo.get_unclosed_events()
        assert len(leftover) == 1
        assert "周末计划" in leftover[0].merged_content

    def test_short_recent_stays_unclosed_under_three_days(
        self, metabolism_service: MetabolismService, config: REMSConfig, fake_llm: FakeLLM,
    ):
        config.unclosed_char_limit = 200
        past = datetime.now() - timedelta(hours=3)
        metabolism_service._repo.save_unclosed_event(UnclosedEvent(
            id="UC-keep-short",
            content_fragments=["短"],
            created_at=past,
            updated_at=past,
            last_hit_time=past,
        ))
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed_indices": [1, 2],
        })
        memory = MemoryExperience(
            subject_id="jshi-1",
            text="然后呢",
            objects={},
            source_ids=("s1",),
            occurred_at=datetime.now(),
        )
        events = metabolism_service.process_input("然后呢", memory=memory)
        assert events == []
        leftover = metabolism_service._repo.get_unclosed_events()
        assert len(leftover) == 1

    def test_recent_same_interlocutor_stays_unclosed(
        self, metabolism_service: MetabolismService, fake_llm: FakeLLM,
    ):
        now = datetime.now()
        metabolism_service._repo.save_unclosed_event(UnclosedEvent(
            id="UC-keep",
            content_fragments=["还在商量出门"],
            interlocutor="OBJ-A",
            created_at=now,
            updated_at=now,
            last_hit_time=now,
        ))
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed_indices": [1, 2],
        })
        memory = MemoryExperience(
            subject_id="jshi-1",
            text="然后呢",
            objects={"甲": "OBJ-A"},
            interlocutor="OBJ-A",
            source_ids=("s1",),
            occurred_at=now,
        )
        events = metabolism_service.process_input("然后呢", memory=memory)
        assert events == []
        leftover = metabolism_service._repo.get_unclosed_events()
        assert len(leftover) == 1
        assert leftover[0].interlocutor == "OBJ-A"
