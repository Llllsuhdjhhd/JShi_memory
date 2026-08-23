"""记忆写入链路（MemoryBackendPort.ingest_batch）单元测试。

覆盖 design/210（输入协议）、410（对象时间线）、610（事件封装）、810（存储）：
- 主体记忆（objects 空）与带对象输入；
- content_raw == 输入 text（原文保全）；
- 角色列表 = 主体恒在 + 对象映射表（不抽取）；
- 事件级主体情感、origin / source_ids 落库；
- stored_marks：input_id → 封存事件 id 列表；
- 对象时间线条目（无情感）。
"""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryInput
from rems.services.event_service import EventService
from rems.services.metabolism_service import MetabolismService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.event_enrichment import EventEnrichmentSkill
from rems.storage.repository import (
    EventRepository,
    MetabolismRepository,
    ObjectTimelineRepository,
    StoredMarksRepository,
)

from ..conftest import FakeLLM


@pytest.fixture()
def memory_pipeline(config, db, fake_llm, vector_store):
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    obj_repo = ObjectTimelineRepository(db)
    marks_repo = StoredMarksRepository(db)
    enrichment_skill = EventEnrichmentSkill(fake_llm, config)
    event_service = EventService(
        config, fake_llm, event_repo, vector_store,
        enrichment_skill, object_timeline_repo=obj_repo,
    )
    boundary_skill = BoundaryDetectionSkill(fake_llm, config)
    metabolism = MetabolismService(config, meta_repo, boundary_skill, event_service)
    pipeline = REMSPipeline(
        config=config, llm=fake_llm, db=db,
        event_repo=event_repo, role_repo=None, meta_repo=meta_repo,
        event_service=event_service, role_service=None,
        metabolism_service=metabolism,
        belief_revision_service=None,
        object_timeline_repo=obj_repo, stored_marks_repo=marks_repo,
    )
    return pipeline, event_repo, obj_repo, marks_repo


class TestIngestBatch:
    def test_single_event_with_objects(
        self, memory_pipeline, fake_llm: FakeLLM
    ):
        pipeline, event_repo, obj_repo, marks_repo = memory_pipeline
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
            "new_unclosed_indices": [],
        })
        fake_llm.push_response({
            "summaries": {"L1": "匠石看到了美丽的落日。"},
            "emotion": {"joy": 0.8},
            "objects": [{"name": "阿明", "summary": "阿明与匠石一起看落日。"}],
            "location": "海边",
        })

        batch = MemoryBatch(
            subject_id="jshi-1",
            inputs=(MemoryInput(
                subject_id="jshi-1",
                text="今天看到美丽的落日",
                objects={"阿明": "OBJ-AMING"},
                source_ids=("FACT-1",),
                input_id="seg-001",
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
        assert ev.source_ids == ["FACT-1"]
        assert ev.emotion is not None
        assert ev.emotion.emotion.joy == pytest.approx(0.8)

        subjects = [r for r in ev.role_list if r.is_subject]
        assert len(subjects) == 1
        assert subjects[0].role_id == "jshi-1"
        assert any(r.role_id == "OBJ-AMING" and not r.is_subject for r in ev.role_list)

        entries = obj_repo.list_by_event(ev.event_id)
        assert len(entries) == 1
        assert entries[0].object_id == "OBJ-AMING"
        assert entries[0].name == "阿明"
        assert entries[0].subject_id == "jshi-1"
        assert marks_repo.get("seg-001") == [ev.event_id]

    def test_subject_memory_no_objects(self, memory_pipeline, fake_llm: FakeLLM):
        pipeline, event_repo, obj_repo, _ = memory_pipeline
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
            inputs=(MemoryInput(
                subject_id="jshi-1",
                text="内心平静",
                input_id="seg-002",
            ),),
        )
        result = pipeline.ingest_batch(batch)

        assert result.errors == []
        assert len(result.sealed_event_ids) == 1
        ev = event_repo.get(result.sealed_event_ids[0])
        assert ev is not None
        assert len(ev.role_list) == 1 and ev.role_list[0].is_subject
        assert ev.emotion is not None and ev.emotion.emotion.joy == pytest.approx(0.3)
        assert obj_repo.list_by_event(ev.event_id) == []
        assert result.role_ids == []
        assert result.stored_marks == {"seg-002": result.sealed_event_ids}

    def test_empty_text_error(self, memory_pipeline):
        pipeline, *_ = memory_pipeline
        batch = MemoryBatch(
            subject_id="jshi-1",
            inputs=(MemoryInput(subject_id="jshi-1", text="   "),),
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
            inputs=(
                MemoryInput(subject_id="jshi-1", text=""),
                MemoryInput(subject_id="jshi-1", text="正常输入", input_id="seg-003"),
            ),
        )
        result = pipeline.ingest_batch(batch)
        assert len(result.errors) == 1
        assert len(result.sealed_event_ids) == 1
        assert result.stored_marks == {"seg-003": result.sealed_event_ids}
