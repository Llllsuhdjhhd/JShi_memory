"""ingest（后台线程）与 recall（主线程）并发安全测试（REMS3 并发要求）。

覆盖：共享嵌入模型 / SQLite(WAL+busy_timeout) / Qdrant 本地向量库 / 领域服务。
验收：慢 ingest 进行中 recall 正常返回；无 database is locked、无嵌入/向量异常；
ingest 最终完成且 stored_marks 正确。
"""

from __future__ import annotations

import threading
import time

import pytest

from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience
from rems.recall import (
    HashEmbedding,
    NullReranker,
    QdrantRecallVectorStore,
    RecallPipeline,
)
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


class SlowEmbedding(HashEmbedding):
    """Hash 嵌入 + 人为延迟，放大 ingest 写索引与 recall 读索引的并发窗口。"""

    def _vec(self, text: str) -> list[float]:
        time.sleep(0.03)
        return super()._vec(text)


class SlowLLM(FakeLLM):
    """慢响应 LLM：模拟 ingest 的 boundary/enrich 耗时，让 ingest 长时间占用后台线程。"""

    def __init__(self, config=None, delay: float = 0.2):
        super().__init__(config)
        self._delay = delay

    def complete_json(self, task_type: str, messages: list[dict], **kwargs):
        time.sleep(self._delay)
        return super().complete_json(task_type, messages, **kwargs)

    def complete(self, task_type: str, messages: list[dict], **kwargs) -> str:
        time.sleep(self._delay)
        return super().complete(task_type, messages, **kwargs)


@pytest.fixture()
def concurrent_pipeline(config, db):
    """完整管线：慢 LLM + 慢嵌入 + 真实 Qdrant 内存向量库（同实例并发）。"""
    config.enable_decoration = False
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    role_repo = RoleRepository(db)
    marks_repo = StoredMarksRepository(db)

    embedding = SlowEmbedding()
    vector_store = QdrantRecallVectorStore("rems_events", url=":memory:")
    recall_pipeline = RecallPipeline(
        embedding, vector_store, event_repo, None, reranker=NullReranker()
    )
    llm = SlowLLM(config, delay=0.2)

    enrichment_skill = EventEnrichmentSkill(llm, config)
    role_skill = RoleExtractionSkill(llm, config)
    role_service = RoleService(config, role_repo, role_skill, llm=llm)
    event_service = EventService(
        config, llm, event_repo, None,
        enrichment_skill, role_service=role_service,
    )
    boundary_skill = BoundaryDetectionSkill(llm, config)
    metabolism = MetabolismService(config, meta_repo, boundary_skill, event_service)
    pipeline = REMSPipeline(
        config=config, llm=llm, db=db,
        event_repo=event_repo, role_repo=role_repo, meta_repo=meta_repo,
        event_service=event_service, role_service=role_service,
        metabolism_service=metabolism, belief_revision_service=None,
        recall_pipeline=recall_pipeline, stored_marks_repo=marks_repo,
    )
    return pipeline, event_repo, marks_repo


def _push_ingest_responses(llm: FakeLLM, l1_text: str) -> None:
    llm.push_response({
        "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
        "new_unclosed_indices": [],
    })
    llm.push_response({
        "summaries": {"L1": l1_text},
        "emotion": {"joy": 0.5},
        "objects": [],
        "location": None,
    })


def _experience(text: str, segment_id: str, source_id: str, objects=None) -> MemoryExperience:
    return MemoryExperience(
        subject_id="jshi-1",
        text=text,
        objects=dict(objects or {}),
        source_ids=(source_id,),
        segment_id=segment_id,
        origin="external",
    )


class TestIngestRecallConcurrency:
    def test_sqlite_wal_enabled(self, config, db):
        from sqlalchemy import text

        with db.engine.connect() as conn:
            mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        assert str(mode).lower() == "wal"

    def test_recall_not_blocked_during_ingest(self, concurrent_pipeline):
        pipeline, event_repo, marks_repo = concurrent_pipeline
        llm: SlowLLM = pipeline.llm

        # 批次 A：同步完成，给 recall 提供可检索内容
        _push_ingest_responses(llm, "今天看到美丽的落日")
        res_a = pipeline.ingest_batch(MemoryBatch(
            subject_id="jshi-1",
            experiences=(_experience("今天看到美丽的落日", "seg-A", "FACT-A"),),
        ))
        assert res_a.errors == []
        assert len(res_a.sealed_event_ids) == 1

        # 批次 B：后台线程慢 ingest（boundary 0.2s + enrich 0.2s + 嵌入 0.03s/次）
        _push_ingest_responses(llm, "和deepseek讨论重构方案")
        batch_b = MemoryBatch(
            subject_id="jshi-1",
            experiences=(_experience(
                "和deepseek讨论重构方案", "seg-B", "FACT-B",
                objects={"deepseek": "OBJ-DEEPSEEK"},
            ),),
        )
        thread_errors: list[BaseException] = []
        result_holder: dict = {}

        def run_ingest() -> None:
            try:
                result_holder["result"] = pipeline.ingest_batch(batch_b)
            except BaseException as exc:  # noqa: BLE001
                thread_errors.append(exc)

        t = threading.Thread(target=run_ingest)
        t.start()
        time.sleep(0.15)  # 确保 ingest 已进入（boundary LLM 正在慢响应）

        # 主线程：ingest 进行中连续 recall，必须立即返回且无异常
        for _ in range(12):
            frags = pipeline.recall("jshi-1", "落日", limit=5)
            assert isinstance(frags, tuple), "recall 未返回片段元组"

        t.join(timeout=120)
        assert not thread_errors, f"ingest 线程异常: {thread_errors}"

        # ingest 最终完成：事件数 2、stored_marks 归属正确
        assert len(event_repo.list_all()) == 2
        res_b = result_holder["result"]
        assert res_b.errors == []
        assert res_b.stored_marks.get("seg-B") == res_b.sealed_event_ids
        assert marks_repo.get("seg-B") == res_b.sealed_event_ids

    def test_concurrent_recall_during_ingest_many(self, concurrent_pipeline):
        """加重并发：后台 ingest 未完成期间主线程跑更多 recall，确保无锁/无异常。"""
        pipeline, event_repo, marks_repo = concurrent_pipeline
        llm: SlowLLM = pipeline.llm

        _push_ingest_responses(llm, "缓存预热事件")
        pipeline.ingest_batch(MemoryBatch(
            subject_id="jshi-1",
            experiences=(_experience("缓存预热事件", "seg-A", "FACT-A"),),
        ))

        _push_ingest_responses(llm, "第二批次事件")
        batch_b = MemoryBatch(
            subject_id="jshi-1",
            experiences=(_experience("第二批次事件", "seg-B", "FACT-B"),),
        )
        thread_errors: list[BaseException] = []

        def run_ingest() -> None:
            try:
                pipeline.ingest_batch(batch_b)
            except BaseException as exc:  # noqa: BLE001
                thread_errors.append(exc)

        t = threading.Thread(target=run_ingest)
        t.start()
        time.sleep(0.1)
        for i in range(20):
            frags = pipeline.recall("jshi-1", f"查询 {i} 缓存预热", limit=3)
            assert isinstance(frags, tuple)
        t.join(timeout=120)
        assert not thread_errors
        assert len(event_repo.list_all()) == 2
