from __future__ import annotations

import json
from pathlib import Path

import pytest

from rems.eval.budget import BudgetExceeded, BudgetLedger
from rems.eval.dataset import questions
from rems.eval.runtime import delivered_memory
from rems.eval.scoring import aggregate, coverage, official_f1, validate_grade
from rems.recall.providers import SentenceTransformerEmbedding


def supported_grade():
    return {"atoms": [{"atom_id": "a1", "status": "direct", "quotes": ["Lee enjoys swimming."],
                       "reasoning": "", "conflict": False, "conflict_quotes": []}],
            "answer_verdict": "correct", "answer_reason": "Matches reference.", "abstains": False}


def test_budget_reserves_before_call_and_survives_restart(tmp_path):
    path = tmp_path / "budget.jsonl"
    ledger = BudgetLedger(path, 0.02)
    reservation = ledger.reserve([{"role": "user", "content": "hello"}], 1000, "core")
    with pytest.raises(BudgetExceeded, match="Unsettled"):
        BudgetLedger(path, 0.02).reserve([], 1, "judge")
    ledger.settle(reservation, {"prompt_tokens": 20, "completion_tokens": 10})
    assert BudgetLedger(path).summary()["known_cost_yuan"] == pytest.approx(0.00012)
    with pytest.raises(BudgetExceeded):
        BudgetLedger(path, 0.001).reserve([], 1000, "judge")


def test_missing_usage_never_releases_money(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl")
    reservation = ledger.reserve([], 100, "core")
    with pytest.raises(RuntimeError, match="Missing API usage"):
        ledger.settle(reservation, None)
    assert ledger.summary()["unsettled"] == 1
    assert ledger.summary()["held_yuan"] > 0


def test_supported_atoms_need_real_delivered_evidence():
    atoms = [{"id": "a1", "claim": "Lee likes swimming"}]
    grade = supported_grade()
    assert validate_grade(grade, atoms, "Lee enjoys swimming.") == grade
    with pytest.raises(ValueError, match="exact quote"):
        validate_grade(grade, atoms, "Wang enjoys swimming.")
    grade["atoms"][0]["quotes"] = []
    with pytest.raises(ValueError, match="requires evidence"):
        validate_grade(grade, atoms, "")


def test_conflict_prevents_complete_and_adversarial_has_no_coverage():
    grade = supported_grade()
    grade["atoms"][0]["conflict"] = True
    assert coverage(grade)["sufficient"] == 1
    assert coverage(grade)["complete"] is False
    assert coverage({"atoms": []})["sufficient"] is None


def test_capture_excludes_question_and_private_full_history():
    prompt = "【活跃区】\nNothing\n\n【回忆】\nM1 Lee enjoys swimming.\n\n【本轮】\nWhat does Lee enjoy?"
    memory = delivered_memory(prompt)
    assert memory == "【回忆】\nM1 Lee enjoys swimming."
    assert "What" not in memory
    assert delivered_memory("【本轮】\nLee enjoys swimming?") == ""


def test_memory_and_answer_judging_are_isolated_and_refusal_not_credited():
    from rems.eval.scoring import judge
    class Judge:
        def __init__(self):
            self.requests = []
        def complete_json(self, task, messages):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {"atoms": [{"atom_id":"a1", "status":"missing", "quotes":[],
                    "reasoning":"", "conflict":False, "conflict_quotes":[]}]}
            return {"answer_verdict":"correct", "answer_reason":"honest refusal", "abstains":True}
    model = Judge()
    result = judge(model, question="What sport?", reference="swimming",
        atoms=[{"id":"a1","claim":"swimming"}], memory="MEMORY_SENTINEL",
        answer="ANSWER_SENTINEL Not mentioned")
    assert result["answer_verdict"] == "wrong"
    assert "ANSWER_SENTINEL" not in json.dumps(model.requests[0])
    assert "MEMORY_SENTINEL" not in json.dumps(model.requests[1])


def test_adversarial_factual_guess_cannot_be_graded_correct():
    from rems.eval.scoring import judge
    class Judge:
        def complete_json(self, *args):
            return {"answer_verdict":"correct", "answer_reason":"plausible", "abstains":False}
    result = judge(Judge(), question="Unknown?", reference="", atoms=[],
                   memory="", answer="Guitar", adversarial=True)
    assert result["answer_verdict"] == "wrong"


def test_macro_metrics_exclude_unscored_and_no_answer_coverage():
    grade = supported_grade()
    rows = [{"status": "scored", "category": 4, "grade": grade,
             "coverage": coverage(grade), "official_f1": 1},
            {"status": "scored", "category": 5,
             "grade": {**grade, "atoms": []}, "coverage": coverage({"atoms": []}), "official_f1": 1},
            {"status": "error", "category": 2}]
    result = aggregate(rows)
    assert result["scored"] == 2 and result["errors"] == 1
    assert result["coverage_questions"] == 1 and result["sufficient"] == 1


def test_stratified_selection_preserves_indices_and_seed():
    sample = {"qa": [{"category": c, "question": str(i)}
                      for i, c in enumerate([4, 4, 4, 2, 1, 5, 3])]}
    picked = questions(sample, 5, 42)
    assert {qa["category"] for _, qa in picked} == {1, 2, 3, 4, 5}
    assert picked == questions(sample, 5, 42)
    assert all(sample["qa"][i] == qa for i, qa in picked)


def test_adversarial_uses_official_options_and_decodes_choice():
    from rems.eval.dataset import render_question, canonical_answer
    qa = {"category":5,"question":"What instrument?","adversarial_answer":"clarinet"}
    text, options = render_question(qa, 42)
    assert "clarinet" in text and "Not mentioned in the conversation" in text
    assert options == render_question(qa, 42)[1]
    correct = next(k for k,v in options.items() if v.startswith("Not mentioned"))
    assert canonical_answer(f"({correct})", options) == "Not mentioned in the conversation"


def test_query_instruction_only_applies_to_queries():
    class FakeVector(list):
        def tolist(self):
            return list(self)
    class FakeModel:
        def __init__(self):
            self.calls = []
        def encode(self, texts, **kwargs):
            self.calls.append(texts)
            return FakeVector([FakeVector([1., 0.]) for _ in texts])
    encoder = SentenceTransformerEmbedding("unused", "search: ")
    model = FakeModel()
    encoder._model = model
    encoder.embed_documents(["fact"])
    encoder.embed_query("question")
    assert model.calls == [["fact"], ["search: question"]]


def test_official_f1_stemming_lists_and_abstention():
    pytest.importorskip("nltk")
    assert official_f1(4, "swimming", "swim") == 1
    assert official_f1(1, "swimming", "swim, hike") == .5
    assert official_f1(5, "Not mentioned", "") == 1


@pytest.mark.parametrize("line_id", [1, 99, True])
def test_judge_citation_recovery_uses_existing_lines_only(line_id):
    from rems.eval.scoring import judge

    class Responses:
        def __init__(self):
            self.calls = 0

        def complete_json(self, task, messages):
            self.calls += 1
            if self.calls == 1:
                return {"atoms": []}  # Invalid first response triggers format recovery.
            if self.calls == 2:
                return {"atoms": [{"atom_id": "a1", "status": "direct",
                    "quote_lines": [line_id], "reasoning": "explicit",
                    "conflict": False, "conflict_lines": []}]}
            return {"answer_verdict": "correct", "answer_reason": "matches", "abstains": False}

    llm = Responses()
    arguments = dict(question="What sport?", reference="swimming",
        atoms=[{"id": "a1", "claim": "Lee enjoys swimming"}],
        memory="Lee enjoys swimming.", answer="swimming")
    if type(line_id) is int and line_id == 1:
        result = judge(llm, **arguments)
        assert result["atoms"][0]["quotes"] == [arguments["memory"]]
        assert result["citation_recovery"] == "numbered_delivered_lines"
    else:
        with pytest.raises(ValueError, match="nonexistent evidence line"):
            judge(llm, **arguments)
        assert llm.calls == 2  # Invalid evidence cannot reach final scoring.


def test_interrupted_call_can_be_accounted_at_full_maximum(tmp_path):
    from rems.eval.budget import BudgetLedger
    ledger = BudgetLedger(tmp_path / "budget.jsonl", 1)
    reservation = ledger.reserve([{"role": "user", "content": "test"}], 100, "core")
    ledger.charge_unknown_at_reserved_maximum(reservation["id"], "network failure; usage unknown")
    summary = ledger.summary()
    assert summary["unsettled"] == 0
    assert summary["known_cost_yuan"] == pytest.approx(reservation["charge"])
    last = json.loads(ledger.path.read_text("utf-8").splitlines()[-1])
    assert last["usage_unknown"] is True
    assert "prompt_tokens" not in last
    with pytest.raises(ValueError):
        ledger.charge_unknown_at_reserved_maximum(reservation["id"], "cannot settle twice")


def test_clock_works_after_loading_real_encoder():
    cache = Path.home()/".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5"
    if not any(cache.glob("snapshots/*/model.safetensors")):
        pytest.skip("English encoder is not cached")
    from sentence_transformers import SentenceTransformer
    from rems.eval.runtime import evaluation_clock
    from datetime import datetime
    import rems.recall.pipeline as recall
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True, device="cpu")
    with evaluation_clock(datetime(2023, 5, 1)):
        assert recall.datetime.now().year == 2023
        assert model.encode(["Lee enjoys swimming."]).shape == (1, 384)


def test_real_jshi_core_receives_only_assembled_memory(tmp_path):
    sibling = Path(__file__).resolve().parents[3] / "JShi"
    if not sibling.is_dir():
        pytest.skip("Sibling JShi checkout is required")
    from rems.eval.runtime import answer_question, close_pipeline, load_jshi, SUBJECT
    load_jshi(sibling)
    from rems.config import REMSConfig
    from rems.pipeline import REMSPipeline
    from rems.models.event import Event
    from rems.llm.provider import LLMProvider
    from datetime import datetime

    cfg = REMSConfig(_env_file=None, subject_id=SUBJECT,
        embedding={"provider": "hash"}, storage={"database_url": "sqlite:///" + (tmp_path/"memory.db").as_posix()},
        llm={"api_key": "offline-test"})
    class FakeLLM(LLMProvider):
        def __init__(self):
            self.messages = []
        def complete(self, task_type, messages, **kwargs):
            assert task_type == "core"
            self.messages.extend(messages)
            return json.dumps({"response_plan": {"mode": "respond", "reason": "answer",
                "items": [{"channel": "verbal", "text": "Swimming", "target_ids": []}]}})
    llm = FakeLLM()
    pipe = REMSPipeline.from_config(cfg, llm=llm)
    # A real stored/indexed event; no gold label is provided to the core API.
    from rems.models.event import EventRoleEntry, Importance
    # Object ownership is set using an actual event role, not an oracle retrieval hint.
    event = Event(subject_id=SUBJECT, content_raw="Lee enjoys swimming. PRIVATE RAW OMITTED",
                  summaries={"L1": "Lee enjoys swimming."}, occurred_at=datetime(2023, 5, 1),
                  role_list=[EventRoleEntry(role_id="person-lee", importance=Importance.S)])
    pipe.event_repo.save(event)
    pipe.recall_pipeline.index_event(event)
    try:
        result = answer_question(pipe, llm, tmp_path/"core", {"Lee": "person-lee"},
            "What sport does Lee enjoy?", datetime(2023, 5, 2), 42)
        assert "Swimming" in result["answer"]
        assert result["as_of"].startswith("2023-05-02")
        assert "Lee enjoys swimming." in result["delivered_memory"]
        assert "PRIVATE RAW OMITTED" not in result["delivered_memory"]
        assert len([m for m in llm.messages if m["role"] == "user"]) == 1
        assert not (tmp_path / ".jshi").exists()
    finally:
        close_pipeline(pipe)
