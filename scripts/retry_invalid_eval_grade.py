"""Retry a malformed grade only; never retry a valid wrong answer to improve scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from locomo_jshi_eval import ROOT, make_config, write_json
from rems.eval.budget import BudgetLedger
from rems.eval.client import EvaluationLLM
from rems.eval.dataset import canonical_answer
from rems.eval.scoring import coverage, judge, official_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("safe_id")
    args = ap.parse_args()
    out = args.out.resolve()
    manifest = json.loads((out/"manifest.json").read_text("utf-8"))
    item = next(q for q in manifest["questions"] if q["safe_id"] == args.safe_id)
    dest = out/"grades"/(args.safe_id+".json")
    previous = json.loads(dest.read_text("utf-8"))
    if previous.get("status") != "error":
        raise ValueError("Only an invalid/error grade may be retried")
    archive = out/"grade_errors"/(args.safe_id+f".attempt-{len(list((out/'grade_errors').glob(args.safe_id+'*')))+1}.json")
    write_json(archive, previous)
    answer = json.loads((out/"answers"/(args.safe_id+".json")).read_text("utf-8"))
    rubric = json.loads((out/"rubrics"/(args.safe_id+".json")).read_text("utf-8"))
    from dotenv import dotenv_values
    cfg = make_config(out/"unused", "not-loaded")
    cfg.llm.api_key = os.environ.get("REMS_LLM__API_KEY") or dotenv_values(ROOT/".env").get("REMS_LLM__API_KEY")
    llm = EvaluationLLM(cfg, BudgetLedger(ROOT/"runs/locomo_budget.jsonl", 10), out/"api_audit.jsonl")
    prediction = canonical_answer(answer["answer"], item["answer_options"])
    grade = judge(llm, question=item["input_question"], reference=item["reference"],
        atoms=rubric["atoms"], memory=answer["delivered_memory"], answer=prediction,
        adversarial=item["category"]==5, as_of=answer.get("as_of"))
    write_json(dest, {"question_id":item["question_id"], "category":item["category"],
        "status":"scored", "grade":grade, "coverage":coverage(grade), "canonical_answer":prediction,
        "official_f1":official_f1(item["category"], prediction, item["reference"]),
        "retry_reason":"Invalid citation/schema only; original grade and API outputs retained",
        "recovery_scoring_sha256":hashlib.sha256((ROOT/"src/rems/eval/scoring.py").read_bytes()).hexdigest()})
    print(json.dumps({"question_id":item["question_id"], "status":"scored", "verdict":grade["answer_verdict"]}))


if __name__ == "__main__":
    main()
