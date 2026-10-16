"""Six fixed grading checks, including answerable refusal and adversarial guesses."""
from pathlib import Path
import json
import os

from locomo_jshi_eval import ROOT, make_config, write_json
from rems.eval.budget import BudgetLedger
from rems.eval.client import EvaluationLLM
from rems.eval.scoring import coverage, judge


def main():
    from dotenv import dotenv_values
    cfg = make_config(ROOT / "runs/unused", "not-loaded")
    cfg.llm.api_key = os.environ.get("REMS_LLM__API_KEY") or dotenv_values(ROOT/".env").get("REMS_LLM__API_KEY")
    out = ROOT / "runs/judge_calibration"
    llm = EvaluationLLM(cfg, BudgetLedger(ROOT/"runs/locomo_budget.jsonl", 10), out/"api_audit.jsonl")
    atom = [{"id":"a1", "claim":"Lee enjoys swimming"}]
    cases = [
        ("answerable_refusal", "What sport does Lee enjoy?", "swimming", atom, "", "Not mentioned", False, "wrong", 0),
        ("supported_answer", "What sport does Lee enjoy?", "swimming", atom, "Lee enjoys swimming.", "Swimming", False, "correct", 1),
        ("adversarial_guess", "What instrument did Lee play at the wedding?", "", [], "Lee plays guitar at home.", "Guitar", True, "wrong", None),
        ("adversarial_abstention", "What instrument did Lee play at the wedding?", "", [], "Lee plays guitar at home.", "Not mentioned", True, "correct", None),
        ("wrong_person", "What sport does Lee enjoy?", "swimming", atom, "Sam enjoys swimming.", "Not mentioned", False, "wrong", 0),
        ("correct_answer_without_recall", "What sport does Lee enjoy?", "swimming", atom, "", "Swimming", False, "correct", 0),
    ]
    results = []
    for name, question, reference, atoms, memory, answer, adversarial, expected, expected_coverage in cases:
        grade = judge(llm, question=question, reference=reference, atoms=atoms, memory=memory,
                      answer=answer, adversarial=adversarial, as_of="2023-10-23T09:55:00+08:00")
        cov = coverage(grade)["sufficient"]
        passed = grade["answer_verdict"] == expected and cov == expected_coverage
        results.append({"case":name, "passed":passed, "expected_verdict":expected,
                        "expected_coverage":expected_coverage, "grade":grade, "coverage":cov})
        print(json.dumps({"case":name,"passed":passed,"verdict":grade["answer_verdict"],"coverage":cov}), flush=True)
    write_json(out/"results.json", results)
    if not all(r["passed"] for r in results):
        raise RuntimeError("Judge calibration failed; do not publish benchmark scores")


if __name__ == "__main__":
    main()
