"""候选去向诊断（recall_candidates）：排名、长度、淘汰原因。

诊断只记录 ``_fill_budget`` 实际访问到的候选，必须证明两件事：
写下来的原因与真实的截断/放不下/选中一致；以及开了诊断不改变召回结果。
"""

from __future__ import annotations

from rems.config import REMSConfig
from rems.eval.evidence import candidate_ranks
from rems.models.event import Event
from rems.pipeline import REMSPipeline
from rems.recall.pipeline import CANDIDATE_REASONS


def _seed(pipeline: REMSPipeline, text: str, l1: str) -> Event:
    event = Event(content_raw=text, subject_id="jshi-1", summaries={"L1": l1})
    pipeline.event_repo.save(event)
    pipeline.recall_pipeline.index_event(event)
    return event


def _pipeline(config: REMSConfig, count: int = 3) -> REMSPipeline:
    config.recall_reinforce_enabled = False
    pipeline = REMSPipeline.from_config(config)
    for index in range(count):
        _seed(pipeline, f"今天看到美丽的落日，第{index}次", f"落日{index}")
    return pipeline


class TestCandidateSink:
    def test_selected_candidates_record_rank_level_and_length(self, config: REMSConfig):
        pipeline = _pipeline(config)
        sink: list = []

        fragments = pipeline.recall("jshi-1", "落日", candidate_sink=sink)

        assert len(sink) == len(fragments) == 3
        assert [row.rank for row in sink] == [1, 2, 3]
        assert all(row.selected and row.reason == "selected" for row in sink)
        assert all(row.representation_level for row in sink)
        assert all(row.chars > 0 for row in sink)
        # 预算逐条递减，且剩余量单调不增
        remaining = [row.budget_remaining for row in sink]
        assert remaining == sorted(remaining, reverse=True)
        assert sum(row.chars for row in sink) <= sink[0].budget_remaining + sink[0].chars

    def test_limit_reports_the_blocked_candidate(self, config: REMSConfig):
        pipeline = _pipeline(config)
        sink: list = []

        fragments = pipeline.recall("jshi-1", "落日", limit=1, candidate_sink=sink)

        assert len(fragments) == 1
        assert [row.reason for row in sink] == ["selected", "limit"]
        assert sink[1].selected is False

    def test_no_room_reports_budget_rejection(self, config: REMSConfig):
        pipeline = _pipeline(config, count=1)
        sink: list = []

        fragments = pipeline.recall("jshi-1", "落日", budget_chars=1, candidate_sink=sink)

        assert fragments == ()
        assert [row.reason for row in sink] == ["no_room"]
        assert sink[0].selected is False and sink[0].chars == 0

    def test_sink_does_not_change_results(self, config: REMSConfig):
        pipeline = _pipeline(config)
        quiet = [fragment.event_id for fragment in pipeline.recall("jshi-1", "落日")]

        sink: list = []
        with_sink = [fragment.event_id
                     for fragment in pipeline.recall("jshi-1", "落日", candidate_sink=sink)]

        assert with_sink == quiet
        assert {row.event_id for row in sink} == set(quiet)

    def test_instance_sink_numbers_successive_calls(self, config: REMSConfig):
        pipeline = _pipeline(config)
        sink: list = []
        pipeline.recall_candidate_sink = sink

        pipeline.recall("jshi-1", "落日")
        pipeline.recall("jshi-1", "落日")

        assert sorted({row.call for row in sink}) == [1, 2]
        assert all(row.reason in CANDIDATE_REASONS for row in sink)


class TestCandidateRanks:
    SAMPLE = {"sample_id": "conv-x"}
    EVENTS = {"e1": {"source_ids": ["conv-x:D1:1"]},
              "e2": {"source_ids": ["conv-x:D2:4"]}}

    def test_rank_of_delivered_gold_material(self):
        candidates = [
            {"call": 1, "rank": 1, "event_id": "e2", "selected": True, "reason": "selected",
             "representation_level": "L1", "chars": 12},
            {"call": 1, "rank": 2, "event_id": "e1", "selected": True, "reason": "selected",
             "representation_level": "L1", "chars": 9},
        ]
        result = candidate_ranks(self.SAMPLE, candidates, self.EVENTS, ["D1:1"])

        assert result["recalled"] and result["best_rank"] == 2
        assert result["selected_rank"] == 2 and result["best_selected"] is True
        assert result["truncated"] is False

    def test_gold_material_outside_budget_still_ranked(self):
        candidates = [
            {"call": 1, "rank": 1, "event_id": "e2", "selected": True, "reason": "selected",
             "representation_level": "L1", "chars": 12},
            {"call": 1, "rank": 2, "event_id": "e1", "selected": False, "reason": "no_room",
             "representation_level": None, "chars": 0},
        ]
        result = candidate_ranks(self.SAMPLE, candidates, self.EVENTS, ["D1:1"])

        assert result["best_rank"] == 2 and result["best_reason"] == "no_room"
        assert result["best_selected"] is False and result["selected_rank"] is None

    def test_absent_and_truncated_are_distinguishable(self):
        reached_cut = candidate_ranks(
            self.SAMPLE,
            [{"call": 1, "rank": 1, "event_id": "e2", "selected": True,
              "reason": "selected", "chars": 5},
             {"call": 1, "rank": 2, "event_id": "e2", "selected": False,
              "reason": "limit", "chars": 0}],
            self.EVENTS, ["D1:1"])
        never_recalled = candidate_ranks(
            self.SAMPLE,
            [{"call": 1, "rank": 1, "event_id": "e2", "selected": True,
              "reason": "selected", "chars": 5}],
            self.EVENTS, ["D1:1"])

        assert reached_cut["recalled"] is False and reached_cut["truncated"] is True
        assert never_recalled["recalled"] is False and never_recalled["truncated"] is False
        assert never_recalled["gold"] == ["D1:1"]
