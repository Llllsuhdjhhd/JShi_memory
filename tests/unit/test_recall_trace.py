"""回忆内容落库（recall_traces）单元测试：输入 + 命中条目可回溯。"""

from __future__ import annotations

from rems.config import REMSConfig
from rems.models.event import Event
from rems.pipeline import REMSPipeline


def _seed(pipeline: REMSPipeline, text: str, l1: str) -> Event:
    ev = Event(content_raw=text, subject_id="jshi-1", summaries={"L1": l1})
    pipeline.event_repo.save(ev)
    pipeline.recall_pipeline.index_event(ev)
    return ev


class TestRecallTrace:
    def test_recall_persists_query_and_items(self, config: REMSConfig):
        pipeline = REMSPipeline.from_config(config)
        ev = _seed(pipeline, "今天看到美丽的落日", "看到落日")

        frags = pipeline.recall("jshi-1", "落日")
        assert len(frags) == 1

        rows = pipeline.recall_trace_repo.list_all(subject_id="jshi-1")
        assert len(rows) == 1
        row = rows[0]
        assert row["query"] == "落日"
        assert row["subject_id"] == "jshi-1"
        assert row["level"] == 1
        assert row["n_items"] == 1
        item = row["items"][0]
        assert item["event_id"] == ev.event_id
        assert item["content"] == "看到落日"
        assert item["summary_level"] == "L1"
        assert item["object_id"] is None  # 无外部对象 → 归属为空，正是需要回溯的点

    def test_recall_trace_disabled_skips_write(self, config: REMSConfig):
        config.recall_trace_enabled = False
        pipeline = REMSPipeline.from_config(config)
        _seed(pipeline, "今天看到美丽的落日", "看到落日")

        pipeline.recall("jshi-1", "落日")
        assert pipeline.recall_trace_repo.list_all(subject_id="jshi-1") == []

    def test_empty_recall_still_records_input(self, config: REMSConfig):
        pipeline = REMSPipeline.from_config(config)
        # 没有可召回的记忆，仍应记录本次输入（便于诊断"召回为空"）
        pipeline.recall("jshi-1", "完全不存在的话题xyz")
        rows = pipeline.recall_trace_repo.list_all(subject_id="jshi-1")
        assert len(rows) == 1
        assert rows[0]["query"] == "完全不存在的话题xyz"
        assert rows[0]["n_items"] == 0
