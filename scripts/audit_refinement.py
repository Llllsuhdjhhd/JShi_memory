"""Read-only audit of a completed staged-memory benchmark, no API calls."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rems.services.refinement_service import utc


def read_json(value, fallback):
    return json.loads(value) if value else fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text("utf-8"))
    path = (args.run / "baselines/c0/rems.db").resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    events = {row["event_id"]: dict(row) for row in connection.execute("SELECT * FROM events")}
    connection.close()
    for event in events.values():
        meta = read_json(event["recall_metadata"], {})
        event["refinement"] = meta.get("refinement") or {}
        event["summaries"] = read_json(event["summaries"], {})
        event["source_events"] = read_json(event["source_events"], [])
        event["source_ids"] = read_json(event["source_ids"], [])
    violations, parent_checks = [], []
    hidden = {key for key, event in events.items() if event["refinement"].get("refinement_hidden_by")}
    for key, event in events.items():
        meta = event["refinement"]
        stage = meta.get("memory_stage")
        if stage == 0:
            if event["summaries"] or read_json(event["emotion"], None) or read_json(event["role_list"], []):
                violations.append([key, "small event was enriched"])
            if len(event["content_raw"]) > manifest["evaluation_config"]["memory_refinement"]["small_event_max_chars"]:
                violations.append([key, "small event exceeds configured cap"])
        if stage is not None and stage > 0:
            children = [events.get(eid) for eid in event["source_events"]]
            if not children or any(child is None for child in children):
                violations.append([key, "missing child"])
                continue
            expected = "\n\n".join(child["content_raw"] if stage == 1 else child["summaries"].get("L1", "")
                                     for child in children)
            exact = event["content_raw"] == expected
            if not exact:
                violations.append([key, "L0 is not exact source concatenation"])
            for child in children:
                child_meta = child["refinement"]
                if key not in child_meta.get("refinement_hidden_by", []):
                    violations.append([key, "child has no ownership link"])
                if child_meta.get("memory_stage") != stage - 1 or child["subject_id"] != event["subject_id"]:
                    violations.append([key, "invalid stage or subject"])
            start = min(utc(datetime.fromisoformat(c["refinement"]["event_time_start"])) for c in children)
            end = max(utc(datetime.fromisoformat(c["refinement"]["event_time_end"])) for c in children)
            if (utc(datetime.fromisoformat(meta["event_time_start"])) != start
                    or utc(datetime.fromisoformat(meta["event_time_end"])) != end):
                violations.append([key, "original time span changed"])
            policy = meta["refinement_policy"]
            age = utc(datetime.fromisoformat(meta["refined_at"])) - end
            if age < timedelta(days=policy["age_days"]):
                violations.append([key, "refined before eligible age"])
            raw_len = len(event["content_raw"])
            limit = min(raw_len, max(int(raw_len * policy["l1_retention_ratio"]),
                                    manifest["evaluation_config"]["summary_fuse_min_chars"]))
            if not event["summaries"].get("L1") or len(event["summaries"]["L1"]) > limit:
                violations.append([key, "invalid L1 budget"])
            units = meta.get("refinement_source_units", [])
            if len(units) != len(children):
                violations.append([key, "missing source-unit ranges"])
            for unit, child in zip(units, children):
                original = child["content_raw"] if stage == 1 else child["summaries"].get("L1", "")
                if event["content_raw"][unit["start_char"]:unit["end_char"]] != original:
                    violations.append([key, "source-unit range mismatch"])
            parent_checks.append({"event_id": key, "stage": stage, "children": event["source_events"],
                                  "exact_L0": exact, "raw_chars": raw_len,
                                  "L1_chars": len(event["summaries"].get("L1", "")), "L1_budget": limit})
    for key in hidden:
        for parent in events[key]["refinement"]["refinement_hidden_by"]:
            if parent not in events or key not in events[parent]["source_events"]:
                violations.append([key, "dangling hidden link"])
    trace_count = 0
    for path in (args.run / "question_states").glob("*/core/recall_traces.jsonl"):
        for line in path.read_text("utf-8").splitlines():
            trace_count += 1
            recalled = json.loads(line).get("recalled_event_ids", [])
            if hidden.intersection(recalled):
                violations.append([path.parent.parent.name, "hidden source returned by ordinary recall"])
    roots = [event for event in events.values() if event["refinement"].get("memory_stage") == 0]
    overlaps = []
    for index, event in enumerate(roots):
        text = "".join(event["content_raw"].split())
        if len(text) < 80:
            continue
        for other in roots[index + 1:]:
            shared = sorted(set(event["source_ids"]).intersection(other["source_ids"]))
            other_text = "".join(other["content_raw"].split())
            if shared and min(len(text), len(other_text)) >= 80 and (text in other_text or other_text in text):
                overlaps.append({"events": [event["event_id"], other["event_id"]],
                                 "shared_source_ids": shared})
    sources = {source for event in roots for source in event["source_ids"]}
    fragment_owners = {}
    for event in roots:
        for identity in event["refinement"].get("source_fragment_ids", []):
            fragment_owners.setdefault(identity, []).append(event["event_id"])
    repeated_fragments = {identity: owners for identity, owners in fragment_owners.items() if len(owners) > 1}
    if repeated_fragments:
        violations.append(["source fragments", "multiple small-event owners"])
    sizes = sorted(len(event["content_raw"]) for event in roots)
    result = {"events": len(events), "stage_counts": dict(Counter(str(e["refinement"].get("memory_stage")) for e in events.values())),
        "visible": len(events) - len(hidden), "hidden": len(hidden), "root_source_ids": len(sources),
        "small_event_chars": {"min": min(sizes, default=0), "median": sizes[len(sizes)//2] if sizes else 0,
                              "max": max(sizes, default=0)},
        "audited_recall_traces": trace_count, "parent_checks": parent_checks, "violations": violations,
        "contained_root_pairs_with_shared_sources": overlaps,
        "source_fragments": len(fragment_owners), "repeated_source_fragments": repeated_fragments,
        "scope": "Structural provenance audit; source IDs and exact text do not prove semantic fidelity of summaries."}
    (args.run / "refinement_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "parent_checks"}, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
