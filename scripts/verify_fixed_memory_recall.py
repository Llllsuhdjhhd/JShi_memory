"""Replay real JShi assembly on an immutable run without any paid model calls."""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from locomo_jshi_eval import ROOT, make_config, make_pipe, read_json, write_json
from rems.eval.runtime import answer_question, close_pipeline, load_jshi
from rems.llm.provider import LLMProvider


class CaptureOnly(LLMProvider):
    def __init__(self):
        pass

    def complete(self, task_type, messages, **kwargs):
        if task_type != "core":
            raise RuntimeError(f"Offline replay attempted a model task: {task_type}")
        return json.dumps({"response_plan": {"mode": "respond", "reason": "capture",
            "items": [{"channel": "verbal", "text": "Capture only", "target_ids": []}]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.out or run / "offline_current_code"
    if out.exists():
        raise RuntimeError("Preserve prior replay; choose a fresh output directory")
    out.mkdir(parents=True)
    manifest = read_json(run / "manifest.json")
    load_jshi(ROOT.parent / "JShi")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(manifest["embedding"], local_files_only=True, device="cpu")
    config = make_config(out, "offline", english_word_lexical=manifest["evaluation_config"]["recall_english_word_lexical"])
    config.recall_event_index_level = manifest["event_index_level"]
    config.memory_refinement.enabled = manifest.get("memory_lifecycle") == "staged"
    results = []
    for order, item in enumerate(manifest["questions"]):
        n = item["safe_id"].split("_")[0]
        baseline = run / "baselines" / n
        info = read_json(baseline / "complete.json")
        previous = read_json(run / "answers" / (item["safe_id"] + ".json"))
        work = out / item["safe_id"]
        shutil.copytree(baseline, work / "memory")
        llm = CaptureOnly()
        pipe = make_pipe(work / "memory", config, llm, encoder)
        try:
            captured = answer_question(pipe, llm, work / "core", info["names"],
                item["input_question"], datetime.fromisoformat(previous["as_of"]), manifest["seed"] + order)
        finally:
            close_pipeline(pipe)
        exact = captured["delivered_memory"] == previous["delivered_memory"]
        results.append({"question_id": item["question_id"], "exact_delivered_memory": exact})
        write_json(work / "capture.json", captured)
        print(item["question_id"], exact, flush=True)
    result = {"questions": len(results), "all_exact": all(r["exact_delivered_memory"] for r in results),
              "paid_calls": 0, "results": results}
    write_json(out / "verification.json", result)
    if not result["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
