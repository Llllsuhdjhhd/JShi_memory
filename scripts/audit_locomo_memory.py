"""Audit a completed evaluation memory without API calls or database writes."""
import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rems.config import REMSConfig
from rems.eval.dataset import sessions
from rems.services.event_service import EventService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.out / "manifest.json").read_text("utf-8"))
    cfg = REMSConfig(_env_file=None, _env_prefix="JSHI_EVAL_UNUSED_", **manifest["evaluation_config"])
    service = object.__new__(EventService)
    service._config = cfg
    path = (args.out / "baselines/c0/rems.db").resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    events = [dict(row) for row in connection.execute("SELECT * FROM events")]
    source_keys, raw_keys = defaultdict(list), defaultdict(list)
    violations = []
    dates = 0
    source_ids = set()
    for event in events:
        raw = event["content_raw"]
        levels = json.loads(event["summaries"])
        sources = json.loads(event["source_ids"])
        source_ids.update(sources)
        raw_keys[raw].append(event["event_id"])
        source_keys[(raw, tuple(sorted(sources)))].append(event["event_id"])
        dates += bool((json.loads(event["recall_metadata"]) or {}).get("source_dates"))
        if cfg.event_min_chars and len(raw) >= cfg.event_min_chars * cfg.event_len_factor:
            violations.append({"event": event["event_id"], "kind": "event_length", "chars": len(raw)})
        budgets = service._compute_budget(len(raw)).summary_level_budgets
        previous = None
        for key, value in sorted(levels.items(), key=lambda item: int(item[0][1:])):
            if len(value) > budgets[key] or (previous is not None and len(value) >= previous):
                violations.append({"event": event["event_id"], "kind": "summary_length", "level": key})
            previous = len(value)
    dataset = json.loads((ROOT.parent / "Locomo-Plus/data/locomo10.json").read_text("utf-8"))
    expected = {f"{dataset[0]['sample_id']}:{turn['dia_id']}"
                for _, _, turns in sessions(dataset[0]) for turn in turns}
    result = {"events": len(events), "max_event_chars": max((len(e["content_raw"]) for e in events), default=0),
        "max_L1_chars": max((len(json.loads(e["summaries"]).get("L1", "")) for e in events), default=0),
        "events_with_occurred_at": sum(bool(e["occurred_at"]) for e in events),
        "events_with_source_dates": dates, "expected_source_turns": len(expected),
        "covered_source_turns": len(source_ids & expected), "missing_source_turns": sorted(expected - source_ids),
        "exact_raw_duplicate_groups": [ids for ids in raw_keys.values() if len(ids) > 1],
        "same_source_duplicate_groups": [ids for ids in source_keys.values() if len(ids) > 1],
        "unclosed_remaining": connection.execute("SELECT count(*) FROM unclosed_events").fetchone()[0],
        "violations": violations,
        "scope": "Source coverage proves delivery of source IDs, not preservation of every semantic fact."}
    (args.out / "memory_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
