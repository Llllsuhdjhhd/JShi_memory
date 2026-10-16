"""Read stored recall traces and immutable events; no API or database mutations."""
import collections
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/locomo_round2_clean"


def connect(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def main():
    with connect(OUT / "baselines/c0/rems.db") as database:
        events = {row["event_id"]: dict(row) for row in database.execute("SELECT * FROM events")}
    grouped = collections.defaultdict(list)
    for event in events.values():
        grouped[event["content_raw"]].append(event["event_id"])
    result = {"event_count": len(events), "unique_raw_count": len(grouped),
        "events_with_occurred_at": sum(bool(e["occurred_at"]) for e in events.values()),
        "exact_duplicate_groups": [ids for ids in grouped.values() if len(ids) > 1], "cases": []}
    for index in (16, 19, 24, 47, 116, 163):
        with connect(OUT / f"question_states/c0_q{index}/memory/rems.db") as database:
            traces = [dict(row) for row in database.execute("SELECT * FROM recall_traces")]
        answer = json.loads((OUT / f"answers/c0_q{index}.json").read_text("utf-8"))
        items = []
        for trace in traces:
            for rank, item in enumerate(json.loads(trace["items"]), 1):
                event = events.get(item["event_id"], {})
                items.append({"rank": rank, **item, "stored_create_time": event.get("create_time"),
                    "stored_occurred_at": event.get("occurred_at"),
                    "stored_raw": event.get("content_raw"), "stored_summaries": event.get("summaries")})
        result["cases"].append({"question_index": index, "answer": answer, "recall_trace": items})
    (OUT / "failure_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: result[key] for key in ("event_count", "unique_raw_count", "events_with_occurred_at")}))


if __name__ == "__main__":
    main()
