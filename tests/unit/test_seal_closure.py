"""事件形成收口：序号划完、封存原因、一句事实白描。"""

from __future__ import annotations

from datetime import datetime

import pytest

from rems.models.event import Event, EventRoleEntry
from rems.models.metabolism import UnclosedEvent
from rems.port import MemoryExperience
from rems.services.event_service import EventService
from rems.services.metabolism_service import MetabolismService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.event_enrichment import EventEnrichmentSkill
from rems.storage.database import ObjectDispositionRecord, ObjectStateRecord, ObjectTraitRecord
from rems.storage.repository import EventRepository, MetabolismRepository

from ..conftest import FakeLLM


@pytest.fixture()
def metabolism_service(config, db, fake_llm, vector_store):
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    skill = EventEnrichmentSkill(fake_llm, config)
    event_service = EventService(config, fake_llm, event_repo, vector_store, skill)
    boundary = BoundaryDetectionSkill(fake_llm, config)
    return MetabolismService(config, meta_repo, boundary, event_service, event_repo=event_repo)


def test_unclaimed_sentence_becomes_one_unclosed_line(config, fake_llm: FakeLLM):
    skill = BoundaryDetectionSkill(fake_llm, config)
    result = skill.parse_response(
        {
            "completed_events": [{"content_raw_indices": [1]}],
            "new_unclosed": [{"indices": [3]}],
        },
        ["甲说周末。", "乙问晚饭。", "甲还没定。"],
        shadow_count=0,
    )
    assert result.completed_events[0].content_raw == "甲说周末。"
    assert len(result.new_unclosed) == 2
    assert result.new_unclosed[0].content == "甲还没定。"
    assert result.new_unclosed[1].content == "乙问晚饭。"


def test_continuation_claims_shadow_indices(config, fake_llm: FakeLLM):
    skill = BoundaryDetectionSkill(fake_llm, config)
    result = skill.parse_response(
        {
            "completed_events": [{
                "content_raw_indices": [1, 2],
                "continuation_of": "UC-old",
            }],
            "new_unclosed": [],
        },
        ["旧句。", "新句。"],
        shadow_count=1,
    )
    assert result.completed_events[0].content_raw == "新句。"
    assert result.new_unclosed == []


def test_interleaved_partition_keeps_omitted_sentence(
    metabolism_service: MetabolismService, fake_llm: FakeLLM,
):
    fake_llm.push_response({
        "completed_events": [],
        "new_unclosed": [{"indices": [1, 3]}],
    })
    events = metabolism_service.process_input("甲说周末。乙问晚饭。甲还没定。")
    assert events == []
    lines = metabolism_service._repo.get_unclosed_events()
    texts = [ue.merged_content for ue in lines]
    assert any("乙问晚饭" in text for text in texts)
    assert any("甲说周末" in text and "甲还没定" in text for text in texts)


def test_continuation_merges_old_line(
    metabolism_service: MetabolismService, fake_llm: FakeLLM,
):
    metabolism_service._repo.save_unclosed_event(UnclosedEvent(
        id="UC-old",
        content_fragments=["旧句。"],
    ))
    fake_llm.push_response({
        "completed_events": [{
            "content_raw_indices": [1, 2],
            "continuation_of": "UC-old",
        }],
        "new_unclosed": [],
    })
    fake_llm.push_response({"summaries": {"L1": "续上了"}, "roles": []})
    events = metabolism_service.process_input("新句。")
    assert len(events) == 1
    assert events[0].seal_reason == "closed"
    assert "旧句。" in events[0].content_raw
    assert "新句。" in events[0].content_raw
    assert metabolism_service._repo.get_unclosed_events() == []


def test_split_prefix_records_split_reason(
    metabolism_service: MetabolismService, fake_llm: FakeLLM,
):
    fake_llm.push_response({
        "completed_events": [{
            "content_raw_indices": [1],
            "is_split_prefix": True,
            "split_id": "SP-1",
        }],
        "new_unclosed": [{"indices": [2], "split_id": "SP-1"}],
    })
    fake_llm.push_response({"summaries": {"L1": "前段"}, "roles": []})
    events = metabolism_service.process_input("前段结束。后段还在。")
    assert len(events) == 1
    assert events[0].seal_reason == "split"
    assert events[0].content_raw == "前段结束。"


def test_truncated_summary_does_not_ask_for_an_ending(config, fake_llm: FakeLLM):
    skill = EventEnrichmentSkill(fake_llm, config)
    fake_llm.push_response({"summaries": {"L1": "还在商量"}, "objects": [], "emotion": {}})
    skill.enrich("我们还在商量。", memory_objects={}, seal_reason="truncated")
    user = fake_llm.calls[-1]["messages"][1]["content"]
    assert "不要补写结局" in user
    assert "不要写成这件事已经结束" in user


def test_content_raw_is_not_cut_at_len_msg(config, db, fake_llm: FakeLLM, vector_store):
    event_repo = EventRepository(db)
    skill = EventEnrichmentSkill(fake_llm, config)
    service = EventService(config, fake_llm, event_repo, vector_store, skill)
    raw = "甲" * (config.len_msg + 40)
    fake_llm.push_response({"summaries": {"L1": "很长"}, "objects": [], "emotion": {}})
    event = service.seal_event(raw, objects={}, subject_id="jshi-1", seal_reason="truncated")
    assert event.content_raw == raw
    assert event.seal_reason == "truncated"
    stored = event_repo.get(event.event_id)
    assert stored is not None
    assert stored.content_raw == raw
    assert stored.seal_reason == "truncated"


def test_white_painting_ignores_personality(config, fake_llm: FakeLLM):
    skill = EventEnrichmentSkill(fake_llm, config)
    fake_llm.push_response({
        "summaries": {"L1": "看落日"},
        "emotion": {"joy": 0.2},
        "objects": [{
            "name": "小明",
            "fact": "我和小明一起看了落日。",
            "l3_decision": "性格急躁",
        }],
    })
    result = skill.enrich("我和小明一起看了落日。", memory_objects={"小明": "OBJ-1"})
    assert result.object_snapshots == {"小明": "我和小明一起看了落日。"}
    assert result.object_white_paintings == {}

    fake_llm.push_response({
        "summaries": {"L1": "空"},
        "objects": [{"name": "小明", "l3_decision": "性格急躁"}],
    })
    empty = skill.enrich("一句。", memory_objects={"小明": "OBJ-1"})
    assert empty.object_snapshots == {}


def test_new_event_raises_dormant_same_object(config, db, fake_llm: FakeLLM, vector_store):
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    skill = EventEnrichmentSkill(fake_llm, config)
    event_service = EventService(config, fake_llm, event_repo, vector_store, skill)
    boundary = BoundaryDetectionSkill(fake_llm, config)
    metabolism = MetabolismService(
        config, meta_repo, boundary, event_service, event_repo=event_repo,
    )
    old = Event(
        content_raw="以前见过",
        subject_id="jshi-1",
        role_list=[EventRoleEntry(role_id="OBJ-A", is_subject=False)],
        forgetting_factor=0.001,
    )
    event_repo.save(old)
    fake_llm.push_response({
        "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
        "new_unclosed": [],
    })
    fake_llm.push_response({
        "summaries": {"L1": "又见面"},
        "emotion": {"joy": 0.4},
        "objects": [{"name": "甲", "fact": "我又见到了甲。"}],
    })
    memory = MemoryExperience(
        subject_id="jshi-1",
        text="我又见到了甲。",
        objects={"甲": "OBJ-A"},
        occurred_at=datetime.now(),
    )
    sealed = metabolism.process_input("我又见到了甲。", memory=memory)
    assert len(sealed) == 1
    assert sealed[0].seal_reason == "closed"
    refreshed = event_repo.get(old.event_id)
    assert refreshed is not None
    assert refreshed.forgetting_factor == pytest.approx(config.forgetting_silence_threshold)


def test_knowledge_tables_exist_and_start_empty(db):
    db.create_tables()
    with db.session() as session:
        assert session.query(ObjectTraitRecord).count() == 0
        assert session.query(ObjectStateRecord).count() == 0
        assert session.query(ObjectDispositionRecord).count() == 0
