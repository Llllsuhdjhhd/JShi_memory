"""记忆写入链路（MemoryBackendPort.ingest_batch）单元测试。

覆盖 design/210（输入协议）、410（对象时间线）、610（事件封装）、810（存储）：
- 主体记忆（objects 空）与带对象输入；
- content_raw == 输入 text（原文保全）；
- 角色列表 = 主体恒在 + 对象映射表（不抽取）；
- 事件级主体情感、origin / source_ids 落库；
- stored_marks：segment_id → 封存事件 id 列表；
- 对象时间线条目（无情感）。
"""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience
from rems.services.event_service import EventService
from rems.services.metabolism_service import MetabolismService
from rems.services.role_service import RoleService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.event_enrichment import EventEnrichmentSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.storage.repository import (
    EventRepository,
    MetabolismRepository,
    RoleRepository,
    StoredMarksRepository,
)

from ..conftest import FakeLLM


@pytest.fixture()
def memory_pipeline(config, db, fake_llm, vector_store):
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    role_repo = RoleRepository(db)
    marks_repo = StoredMarksRepository(db)
    enrichment_skill = EventEnrichmentSkill(fake_llm, config)
    role_skill = RoleExtractionSkill(fake_llm, config)
    role_service = RoleService(config, role_repo, role_skill, llm=fake_llm)
    event_service = EventService(
        config, fake_llm, event_repo, vector_store,
        enrichment_skill, role_service=role_service,
    )
    boundary_skill = BoundaryDetectionSkill(fake_llm, config)
    metabolism = MetabolismService(config, meta_repo, boundary_skill, event_service)
    pipeline = REMSPipeline(
        config=config, llm=fake_llm, db=db,
        event_repo=event_repo, role_repo=role_repo, meta_repo=meta_repo,
        event_service=event_service, role_service=role_service,
        metabolism_service=metabolism,
        belief_revision_service=None,
        stored_marks_repo=marks_repo,
    )
    return pipeline, event_repo, role_repo, marks_repo


class TestIngestBatch:
    def test_single_event_with_objects(
        self, memory_pipeline, fake_llm: FakeLLM
    ):
        pipeline, event_repo, role_repo, marks_repo = memory_pipeline
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({
            "summaries": {"L1": "匠石看到了美丽的落日。"},
            "emotion": {"joy": 0.8},
            "objects": [{
                "name": "阿明",
                "fact": "我和小明一起看了落日。",
                "l3_decision": "性格急躁",
            }],
            "location": "海边",
        })

        batch = MemoryBatch(
            subject_id="jshi-1",
            experiences=(MemoryExperience(
                subject_id="jshi-1",
                text="今天看到美丽的落日",
                objects={"阿明": "OBJ-AMING"},
                interlocutor="OBJ-AMING",
                source_ids=("FACT-1",),
                segment_id="seg-001",
                origin="external",
            ),),
        )
        result = pipeline.ingest_batch(batch)

        assert result.errors == []
        assert len(result.sealed_event_ids) == 1
        assert result.stored_marks == {"seg-001": result.sealed_event_ids}
        assert result.role_ids == ["OBJ-AMING"]

        ev = event_repo.get(result.sealed_event_ids[0])
        assert ev is not None
        assert ev.content_raw == "今天看到美丽的落日"
        assert ev.subject_id == "jshi-1"
        assert ev.origin == "external"
        assert ev.seal_reason == "closed"
        assert ev.source_ids == ["FACT-1"]
        assert [
            (item.segment_id, item.object_id)
            for item in ev.interlocutor_attributions
        ] == [("seg-001", "OBJ-AMING")]
        assert ev.emotion is not None
        assert ev.emotion.emotion.joy == pytest.approx(0.8)

        subjects = [r for r in ev.role_list if r.is_subject]
        assert len(subjects) == 1
        assert subjects[0].role_id == "jshi-1"
        obj_entry = next(r for r in ev.role_list if r.role_id == "OBJ-AMING" and not r.is_subject)
        assert obj_entry.role_snapshot.l1_mention is None
        assert obj_entry.role_snapshot.l2_interaction is None
        assert obj_entry.role_snapshot.l3_decision is None

        wps = role_repo.get_white_painting("OBJ-AMING")
        assert len(wps) == 1
        assert wps[0].event_id == ev.event_id
        assert wps[0].subject_id == "jshi-1"
        assert wps[0].role_summary == "我和小明一起看了落日。"
        assert wps[0].l1_mention is None
        assert wps[0].l2_interaction is None
        assert wps[0].l3_decision is None
        assert "性格" not in wps[0].role_summary
        assert marks_repo.get("seg-001") == [ev.event_id]

    def test_merged_segments_keep_each_historical_interlocutor(
        self, memory_pipeline, fake_llm: FakeLLM
    ):
        pipeline, event_repo, _role_repo, _marks_repo = memory_pipeline
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1, 2], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({
            "summaries": {"L1": "两段经历"},
            "emotion": {"joy": 0.2},
            "objects": [],
        })
        batch = MemoryBatch(
            subject_id="jshi-1",
            experiences=(
                MemoryExperience(
                    subject_id="jshi-1", text="和甲商量计划。",
                    objects={"甲": "OBJ-A"}, interlocutor="OBJ-A",
                    segment_id="seg-a",
                ),
                MemoryExperience(
                    subject_id="jshi-1", text="又和乙确认时间。",
                    objects={"乙": "OBJ-B"}, interlocutor="OBJ-B",
                    segment_id="seg-b",
                ),
            ),
        )

        result = pipeline.ingest_batch(batch)

        assert result.errors == []
        assert len(result.sealed_event_ids) == 1
        event = event_repo.get(result.sealed_event_ids[0])
        assert event is not None
        assert [
            (item.segment_id, item.object_id)
            for item in event.interlocutor_attributions
        ] == [("seg-a", "OBJ-A"), ("seg-b", "OBJ-B")]

    def test_unclosed_event_keeps_interlocutors_across_ingest_batches(
        self, memory_pipeline, fake_llm: FakeLLM
    ):
        pipeline, event_repo, _role_repo, _marks_repo = memory_pipeline
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed_indices": [1],
        })
        first = pipeline.ingest_batch(MemoryBatch(
            subject_id="jshi-1",
            experiences=(MemoryExperience(
                subject_id="jshi-1",
                text="约甲见面，但日期还没定。",
                objects={"甲": "OBJ-A"},
                interlocutor="OBJ-A",
                segment_id="seg-a",
            ),),
        ))
        assert first.sealed_event_ids == []
        unclosed = pipeline.meta_repo.get_unclosed_events()[0]
        assert [(item.segment_id, item.object_id) for item in unclosed.interlocutor_attributions] == [
            ("seg-a", "OBJ-A"),
        ]

        fake_llm.push_response({
            "completed_events": [{
                "content_raw_indices": [1, 2],
                "continuation_of": unclosed.id,
            }],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({
            "summaries": {"L1": "约好周六见面"},
            "emotion": {"joy": 0.1},
            "objects": [],
        })
        second = pipeline.ingest_batch(MemoryBatch(
            subject_id="jshi-1",
            experiences=(MemoryExperience(
                subject_id="jshi-1",
                text="后来和乙确认了周六见面。",
                objects={"乙": "OBJ-B"},
                interlocutor="OBJ-B",
                segment_id="seg-b",
            ),),
        ))

        assert second.errors == []
        event = event_repo.get(second.sealed_event_ids[0])
        assert event is not None
        assert {
            (item.segment_id, item.object_id)
            for item in event.interlocutor_attributions
        } == {("seg-a", "OBJ-A"), ("seg-b", "OBJ-B")}

    def test_subject_memory_no_objects(self, memory_pipeline, fake_llm: FakeLLM):
        pipeline, event_repo, role_repo, _ = memory_pipeline
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({
            "summaries": {"L1": "内心平静。"},
            "emotion": {"joy": 0.3},
            "objects": [],
        })

        batch = MemoryBatch(
            subject_id="jshi-1",
            experiences=(MemoryExperience(
                subject_id="jshi-1",
                text="内心平静",
                segment_id="seg-002",
            ),),
        )
        result = pipeline.ingest_batch(batch)

        assert result.errors == []
        assert len(result.sealed_event_ids) == 1
        ev = event_repo.get(result.sealed_event_ids[0])
        assert ev is not None
        assert len(ev.role_list) == 1 and ev.role_list[0].is_subject
        assert ev.emotion is not None and ev.emotion.emotion.joy == pytest.approx(0.3)
        # 主体不写对象白描
        assert role_repo.get_white_painting("jshi-1") == []
        assert result.role_ids == []
        assert result.stored_marks == {"seg-002": result.sealed_event_ids}

    def test_empty_text_error(self, memory_pipeline):
        pipeline, *_ = memory_pipeline
        batch = MemoryBatch(
            subject_id="jshi-1",
            experiences=(MemoryExperience(subject_id="jshi-1", text="   "),),
        )
        result = pipeline.ingest_batch(batch)
        assert result.sealed_event_ids == []
        assert len(result.errors) == 1
        assert "empty" in result.errors[0]

    def test_failed_input_isolated(self, memory_pipeline, fake_llm: FakeLLM):
        """单条失败进 errors，不影响批次内其他 input。"""
        pipeline, *_ = memory_pipeline
        # 第一条：空文本（失败）；第二条：正常封存。
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({
            "summaries": {"L1": "好。"},
            "emotion": {"joy": 0.2},
            "objects": [],
        })
        batch = MemoryBatch(
            subject_id="jshi-1",
            experiences=(
                MemoryExperience(subject_id="jshi-1", text=""),
                MemoryExperience(subject_id="jshi-1", text="正常输入", segment_id="seg-003"),
            ),
        )
        result = pipeline.ingest_batch(batch)
        assert len(result.errors) == 1
        assert len(result.sealed_event_ids) == 1
        assert result.stored_marks == {"seg-003": result.sealed_event_ids}
