"""Compare actual JShi assembly on copies of one fixed memory, without API calls."""
import json
import shutil
from datetime import datetime
from pathlib import Path

from locomo_jshi_eval import ROOT, make_config, make_pipe, read_json, write_json
from rems.eval.runtime import answer_question, close_pipeline, load_jshi
from rems.llm.provider import LLMProvider
from sentence_transformers import SentenceTransformer


class CaptureOnly(LLMProvider):
    def __init__(self):
        pass

    def complete(self, task_type, messages, **kwargs):
        assert task_type == "core"
        return json.dumps({"response_plan": {"mode": "respond", "reason": "capture",
            "items": [{"channel": "verbal", "text": "Capture only", "target_ids": []}]}})


def main():
    source = ROOT / "runs/locomo_round1"
    out = ROOT / "runs/english_recall_fixed_baseline_v2"
    load_jshi(ROOT.parent / "JShi")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
    manifest = read_json(source / "manifest.json")
    names = {"Caroline": "conv-26:speaker_a", "Melanie": "conv-26:speaker_b"}
    records = []
    for index in (19, 130, 102, 12, 24, 47, 116, 16):
        item = next(q for q in manifest["questions"] if q["index"] == index)
        previous = read_json(source / "answers" / (item["safe_id"] + ".json"))
        for enabled in (False, True):
            directory = out / (item["safe_id"] + ("_words" if enabled else "_bigrams"))
            if directory.exists():
                raise RuntimeError(f"Preserve existing capture: {directory}")
            shutil.copytree(source / "baselines/c0", directory)
            cfg = make_config(directory, "offline", english_word_lexical=enabled)
            pipe = make_pipe(directory, cfg, CaptureOnly(), encoder)
            try:
                result = answer_question(pipe, CaptureOnly(), directory / "core", names,
                    item["input_question"], datetime.fromisoformat(previous["as_of"]),
                    manifest["seed"] + manifest["questions"].index(item))
            finally:
                close_pipeline(pipe)
            records.append({"question_id": item["question_id"], "question": item["question"],
                "english_word_lexical": enabled, "delivered_memory": result["delivered_memory"],
                "sources": result["sources"]})
            print(item["question_id"], enabled, flush=True)
    write_json(out / "comparison.json", records)


if __name__ == "__main__":
    main()
