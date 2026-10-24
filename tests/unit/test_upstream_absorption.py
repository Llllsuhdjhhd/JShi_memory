"""Regression checks for capabilities absorbed onto the local baseline."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rems.config import REMSConfig
from rems.eval.evidence import retrieval_evidence
from rems.llm.provider import LLMProvider
from rems.models.event import Event
from rems.pipeline import REMSPipeline


@pytest.mark.parametrize("cache", [{"prompt_cache_hit_tokens": 0},
    {"prompt_tokens_details": SimpleNamespace(cached_tokens=7)}, {}])
def test_provider_options_and_cache_accounting(cache):
    cfg = REMSConfig(_env_file=None, llm={"api_key": "offline-test",
        "extra_body": {"thinking": {"type": "disabled"}}})
    provider = LLMProvider(cfg)
    provider._client = Mock()
    provider._client.chat.completions.create.return_value = SimpleNamespace(
        model="actual-model", choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12, **cache))
    assert provider.complete("summary", []) == "ok"
    assert provider._client.chat.completions.create.call_args.kwargs["extra_body"] == cfg.llm.extra_body
    metric = provider.invocation_history()[0]
    assert metric.response_model == "actual-model"
    assert metric.cached_tokens == (0 if "prompt_cache_hit_tokens" in cache else 7 if cache else None)
    assert metric.total_tokens == 12


def test_default_provider_omits_extra_body():
    provider = LLMProvider(REMSConfig(_env_file=None, llm={"api_key": "offline-test"}))
    provider._client = Mock()
    provider._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    provider.complete("summary", [])
    assert "extra_body" not in provider._client.chat.completions.create.call_args.kwargs
    assert provider.invocation_history()[0].cached_tokens is None


def test_recall_can_disable_and_explicitly_override_reinforcement(config):
    config.recall_reinforce_enabled = False
    pipe = REMSPipeline.from_config(config)
    event = Event(subject_id="stone", content_raw="今天看到落日", summaries={"L1": "看到落日"})
    pipe.event_repo.save(event)
    pipe.recall_pipeline.index_event(event)
    before = pipe.event_repo.get(event.event_id).forgetting_factor
    try:
        assert pipe.recall("stone", "落日")
        assert pipe.event_repo.get(event.event_id).forgetting_factor == before
        assert pipe.recall("stone", "落日", reinforce=True)
        reinforced = pipe.event_repo.get(event.event_id).forgetting_factor
        assert reinforced > before
        assert pipe.recall("stone", "落日", reinforce=False)
        assert pipe.event_repo.get(event.event_id).forgetting_factor == reinforced
    finally:
        pipe.recall_pipeline._vector_store._client.close()
        pipe.db.engine.dispose()


SAMPLE = {"sample_id": "conv-x", "conversation": {
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "I enjoy swimming."},
                  {"speaker": "B", "dia_id": "D1:2", "text": "I prefer hiking."}],
    "session_2_date_time": "1:56 pm on 9 May, 2023",
    "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "Back again."}]}}


def event(**kwargs):
    return {"event_id": "E1", "source_ids": ["conv-x:D1:1", "conv-x:D1:2"],
            "content_raw": "A: I enjoy swimming.", **kwargs}


def test_turn_does_not_fall_back_to_chunk():
    result = retrieval_evidence(SAMPLE, [event()], ["D1:2"])
    assert result["session_hit"] and result["chunk_hit"]
    assert result["turn_hit"] is False
    assert result["events"][0]["turn_dias"] == ["D1:1"]
    summary = retrieval_evidence(SAMPLE, [event(content_raw="A enjoys swimming")], ["D1:1"])
    assert summary["turn_recall"] == 0


def test_real_jshi_fullwidth_speaker_prefix_matches():
    result = retrieval_evidence(SAMPLE, [event(content_raw="A：I enjoy swimming.")], ["D1:1"])
    assert result["turn_recall"] == 1


def test_foreign_sources_cannot_match_by_time_or_raw_text():
    result = retrieval_evidence(SAMPLE, [event(source_ids=["other:D1:1"],
        occurred_at=datetime(2023, 5, 8, 13, 56).isoformat())], ["D1:1"])
    assert result["cross_conversation_events"] == 1
    assert not any(result[level + "_hit"] for level in ("session", "chunk", "turn"))


def test_sources_cover_multiple_sessions_despite_single_timestamp():
    result = retrieval_evidence(SAMPLE, [event(source_ids=["conv-x:D1:1", "conv-x:D2:1"],
        content_raw="B: Back again.", occurred_at="2023-05-08T13:56:00")], ["D2:1"])
    assert result["session_recall"] == result["chunk_recall"] == result["turn_recall"] == 1


def test_timestamp_fallback_does_not_claim_chunk_provenance():
    result = retrieval_evidence(SAMPLE, [event(source_ids=[], occurred_at="2023-05-08T13:58:00")], ["D1:1"])
    assert result["session_hit"] and result["turn_hit"]
    assert result["chunk_hit"] is False
    assert result["events"][0]["method"] == "time"
    assert retrieval_evidence(SAMPLE, [], [])["turn_recall"] is None


def test_bare_ids_stay_unmatched_in_namespaced_runner():
    result = retrieval_evidence(SAMPLE, [event(source_ids=["D1:1"],
        occurred_at="2023-05-08T13:56:00")], ["D1:1"])
    assert result["unmatched_events"] == 1
    assert result["session_recall"] == 0


def test_report_keeps_legacy_results_and_reports_new_diagnostics(tmp_path):
    import json
    import runpy
    from pathlib import Path
    from rems.eval.budget import BudgetLedger

    runner = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/locomo_jshi_eval.py"))
    assert runner["make_config"](tmp_path, "offline").recall_reinforce_enabled is False
    diagnostic = retrieval_evidence(SAMPLE, [event()], ["D1:2"])
    grade = {"status": "scored", "category": 4, "official_f1": 1,
        "grade": {"answer_verdict": "correct", "atoms": []},
        "coverage": {"direct": None, "sufficient": None, "complete": None, "conflict": None}}
    manifest = {"scope": "offline-test", "questions": [
        {"safe_id": "q1", "question_id": "x:q1", "question": "Q?", "category": 4},
        {"safe_id": "q2", "question_id": "x:q2", "question": "Q2?", "category": 4}]}
    runner["write_json"](tmp_path / "grades/q1.json", {**grade, "retrieval_evidence": diagnostic})
    runner["write_json"](tmp_path / "grades/q2.json", grade)
    summary = runner["report"](tmp_path, manifest, BudgetLedger(tmp_path / "ledger.jsonl", 1))
    assert summary["scored"] == 2
    assert summary["retrieval_evidence"]["questions"] == 1
    assert summary["retrieval_evidence"]["chunk_hit_rate"] == 1
    assert summary["retrieval_evidence"]["turn_hit_rate"] == 0
    assert "召回证据诊断" in (tmp_path / "report.md").read_text("utf-8")
    assert json.loads((tmp_path / "summary.json").read_text("utf-8")) == summary


def test_saved_trace_audit_excludes_baseline_and_never_changes_databases(tmp_path):
    import hashlib
    import json
    import runpy
    import sqlite3
    from pathlib import Path
    from contextlib import closing

    tool = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/audit_locomo_retrieval.py"))
    baseline, question = tmp_path / "baseline.db", tmp_path / "question.db"
    for path in (baseline, question):
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE TABLE recall_traces (recall_id TEXT, items TEXT)")
            db.execute("CREATE TABLE events (event_id TEXT, source_ids TEXT, occurred_at TEXT, content_raw TEXT)")
            db.execute("INSERT INTO recall_traces VALUES (?, ?)", ("old", json.dumps([{"event_id": "old-event"}])))
            db.commit()
    assert tool["saved_events"](question, baseline) is None
    with closing(sqlite3.connect(question)) as db:
        db.execute("INSERT INTO recall_traces VALUES (?, ?)", ("new", json.dumps([{"event_id": "new-event"}])))
        db.execute("INSERT INTO events VALUES (?, ?, ?, ?)",
                   ("new-event", '["conv-x:D1:1"]', "2023-05-08T13:56:00", "A: I enjoy swimming."))
        db.commit()
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (baseline, question)]
    actual = tool["saved_events"](question, baseline)
    assert [item["event_id"] for item in actual] == ["new-event"]
    assert actual[0]["source_ids"] == ["conv-x:D1:1"]
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (baseline, question)]
