"""Diagnose a frozen run's retrieval using saved SQLite traces, without API calls.

Writes a separate report; original answers, grades and manifests remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rems.eval.dataset import read_dataset
from rems.eval.evidence import candidate_ranks, retrieval_evidence


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def saved_events(database, baseline):
    """Exclude traces copied from history, even when timestamps coincide."""
    with closing(sqlite3.connect(baseline.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        historical = {row[0] for row in connection.execute("SELECT recall_id FROM recall_traces")}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        traces = [row for row in connection.execute("SELECT * FROM recall_traces")
                  if row["recall_id"] not in historical]
        if not traces:
            return None  # Missing instrumentation is not evidence of zero recall.
        ids = {item["event_id"] for row in traces for item in json.loads(row["items"])}
        events = []
        for identity in sorted(ids):
            row = connection.execute("SELECT event_id, source_ids, occurred_at, content_raw "
                                     "FROM events WHERE event_id = ?", (identity,)).fetchone()
            if row is None:
                raise ValueError(f"Trace references a missing event: {identity}")
            events.append({**dict(row), "source_ids": json.loads(row["source_ids"] or "[]")})
        return events


def events_by_ids(database, ids):
    """按 id 读事件：候选里有从未交付、因此不在 traces 里的条目。"""
    if not ids:
        return {}
    out = {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for identity in sorted(ids):
            row = connection.execute("SELECT event_id, source_ids, occurred_at, content_raw "
                                     "FROM events WHERE event_id = ?", (identity,)).fetchone()
            if row is not None:
                out[identity] = {**dict(row), "source_ids": json.loads(row["source_ids"] or "[]")}
    return out


def audit(run, data_path):
    manifest = read_json(run / "manifest.json")
    data, digest = read_dataset(data_path)
    if digest != manifest["dataset_sha256"]:
        raise ValueError("Dataset differs from frozen manifest")
    samples = {sample["sample_id"]: sample for sample in data}
    records, unavailable = [], []
    for item in manifest["questions"]:
        name = item["safe_id"]
        baseline = run / "baselines" / name.split("_")[0] / "rems.db"
        database = run / "question_states" / name / "memory/rems.db"
        if not database.is_file() or not baseline.is_file():
            unavailable.append(item["question_id"])
            continue
        events = saved_events(database, baseline)
        if events is None:
            unavailable.append(item["question_id"])
            continue
        grade_path = run / "grades" / (name + ".json")
        grade = read_json(grade_path) if grade_path.is_file() else {}
        answer_path = run / "answers" / (name + ".json")
        candidates = (read_json(answer_path).get("recall_candidates")
                      if answer_path.is_file() else None)
        gold_candidate = None
        if candidates:
            known = {event["event_id"]: event for event in events}
            known.update(events_by_ids(database, {row["event_id"] for row in candidates} - known.keys()))
            gold_candidate = candidate_ranks(samples[item["sample_id"]], candidates, known,
                                             item.get("evidence", []))
        records.append({"question_id": item["question_id"], "question": item["question"],
            "category": item["category"], "retrieved_event_count": len(events),
            "answer_verdict": grade.get("grade", {}).get("answer_verdict"),
            "delivered_sufficient": grade.get("coverage", {}).get("sufficient"),
            **({"gold_candidate": gold_candidate} if gold_candidate else {}),
            **retrieval_evidence(samples[item["sample_id"]], events, item.get("evidence", []))})
    eligible = [row for row in records if row["evidence"]]
    def mean(key):
        return sum(row[key] for row in eligible) / len(eligible) if eligible else None
    def median(values):
        values = sorted(values)
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    ranked = [row["gold_candidate"] for row in records if row.get("gold_candidate")]
    gold_material = None
    if ranked:
        gold_material = {
            "questions": len(ranked),
            "recalled": sum(row["recalled"] for row in ranked),
            "delivered": sum(bool(row.get("best_selected")) for row in ranked),
            "not_recalled": sum(not row["recalled"] for row in ranked),
            "truncated": sum(row["truncated"] for row in ranked),
            "best_rank_median": median([row["best_rank"] for row in ranked if row["recalled"]]),
            "delivered_rank_median": median([row["selected_rank"] for row in ranked
                                             if row.get("selected_rank") is not None]),
            "best_reason_counts": dict(Counter(row.get("best_reason") for row in ranked
                                               if row["recalled"])),
        }
    return {"source_run": str(run.resolve()),
        "scope": "saved retrieved events + backend candidates; original raw grades",
        "questions": len(manifest["questions"]), "diagnosed": len(records),
        "unavailable": unavailable, "with_evidence": len(eligible),
        **{level + "_hit_rate": mean(level + "_hit") for level in ("session", "chunk", "turn")},
        **{level + "_recall": mean(level + "_recall") for level in ("session", "chunk", "turn")},
        "cross_conversation_events": sum(row["cross_conversation_events"] for row in records),
        "wrong_despite_turn_hit": [row["question_id"] for row in eligible
                                   if row["turn_hit"] and row["answer_verdict"] == "wrong"],
        **({"gold_material": gold_material} if gold_material else {}),
        "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT.parent / "Locomo-Plus/data/locomo10.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a new diagnostic report")
    result = audit(args.run, args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
