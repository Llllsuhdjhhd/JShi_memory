#!/usr/bin/env python3
"""Rollback one chunk ingest from a scenario workspace (best-effort)."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rems.config import REMSConfig, StorageConfig
from rems.storage.database import Database
from rems.storage.repository import EventRepository
from rems.storage.vector_store import VectorStore
from rems.embedding.tri_band import TriBandEncoder
from rems.storage.repository import RoleRepository
from tests.scenarios.common.harness import find_repo_root, ScenarioWorkspace


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback last chunk ingest artifacts")
    parser.add_argument("chunk_id", type=int)
    parser.add_argument("--run-name", default="continuous_run")
    parser.add_argument(
        "--shadow-chars",
        type=int,
        default=None,
        help="Target merged unclosed shadow length after rollback (from prior debug shadow_load)",
    )
    args = parser.parse_args()

    repo = find_repo_root()
    ws = ScenarioWorkspace.create(
        repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name,
        run_name=args.run_name,
    )
    debug_dir = ws.chunk_debug_dir(args.chunk_id)
    pipe_path = debug_dir / "pipeline_trace.json"
    if not pipe_path.is_file():
        print(f"Missing {pipe_path}", file=sys.stderr)
        return 1

    pipe = json.loads(pipe_path.read_text(encoding="utf-8"))
    sealed = next(s.get("sealed") or [] for s in pipe["steps"] if s["step"] == "metabolism_seal")
    event_ids = [x["event_id"] for x in sealed if x.get("event_id")]
    if not event_ids:
        print("No sealed events in pipeline trace", file=sys.stderr)
        return 1

    shadow_load = next(s for s in pipe["steps"] if s["step"] == "shadow_load")
    target_shadow = args.shadow_chars if args.shadow_chars is not None else int(
        shadow_load.get("shadow_chars") or 0
    )

    recall_step = next(s for s in pipe["steps"] if s["step"] == "recall_log_append")
    recall_id = recall_step.get("recall_id")

    db_path = ws.db_path
    conn = sqlite3.connect(db_path)

    for eid in event_ids:
        conn.execute("DELETE FROM white_painting_entries WHERE event_id=?", (eid,))
        conn.execute("DELETE FROM event_tier1 WHERE event_id=?", (eid,))
        conn.execute("DELETE FROM events WHERE event_id=?", (eid,))
    if recall_id:
        conn.execute("DELETE FROM recall_log WHERE recall_id=?", (recall_id,))

    rows = conn.execute("SELECT id, content_fragments FROM unclosed_events").fetchall()
    for uid, frags_json in rows:
        frags = json.loads(frags_json or "[]")
        merged = "".join(frags)
        if target_shadow > 0 and len(merged) > target_shadow:
            trimmed = merged[:target_shadow]
            conn.execute(
                "UPDATE unclosed_events SET content_fragments=? WHERE id=?",
                (json.dumps([trimmed], ensure_ascii=False), uid),
            )
    conn.commit()
    conn.close()

    cfg = REMSConfig(storage=StorageConfig(
        database_url=f"sqlite:///{db_path.as_posix()}",
        qdrant_path=str(ws.qdrant_path),
    ))
    vector = VectorStore(cfg)
    for eid in event_ids:
        vector.delete_event(eid)

    prev_chunk = args.chunk_id - 1
    ws.save_last_chunk_idx(prev_chunk, extra={"last_debug_dir": f"debug/chunk_{prev_chunk}"})

    print(f"Rollback chunk {args.chunk_id}: removed {len(event_ids)} events, recall_id={recall_id}")
    print(f"  events deleted: {event_ids}")
    print(f"  last_chunk_idx -> {prev_chunk}, unclosed shadow target {target_shadow} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
