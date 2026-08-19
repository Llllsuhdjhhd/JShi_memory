#!/usr/bin/env python3
"""Audit recall quality for one ingested chunk (UTF-8 safe on Windows).

Usage (from repo root)::

    python tests/scenarios/hongloumeng/audit_recall.py 71
    python tests/scenarios/hongloumeng/audit_recall.py 71 --run-name continuous_run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rems.observability.console_io import configure_stdio_utf8

configure_stdio_utf8()

from tests.scenarios.common.harness import find_repo_root
from tests.scenarios.hongloumeng.dataset import get_chunk


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit recall block for one chunk")
    parser.add_argument("chunk_id", type=int, help="Chunk id (e.g. 71)")
    parser.add_argument("--run-name", default="continuous_run")
    parser.add_argument("--top", type=int, default=15, help="How many recall items to list")
    args = parser.parse_args()

    repo = find_repo_root()
    debug_dir = (
        repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name
        / "debug" / f"chunk_{args.chunk_id}"
    )
    if not debug_dir.is_dir():
        print(f"找不到 debug 目录: {debug_dir}", file=sys.stderr)
        return 1

    chunk = get_chunk(args.chunk_id)
    print(f"=== Chunk {args.chunk_id} 主题（前 500 字）===")
    print(chunk.content[:500].replace("\n", " "))
    print("...")
    print()

    trace_path = debug_dir / "recall_trace.json"
    pipe_path = debug_dir / "pipeline_trace.json"
    if trace_path.is_file():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        for entry in trace.get("query_act", {}).get("entries", []):
            act = entry.get("act_text") or ""
            print(f"=== act_query ({len(act)} 字, source={entry.get('source')}) ===")
            print(act)
            print()

    if not pipe_path.is_file():
        print(f"缺少 {pipe_path}")
        return 1

    pipe = json.loads(pipe_path.read_text(encoding="utf-8"))
    recall_ids = next(
        s["event_ids"] for s in pipe["steps"] if s["step"] == "recall_build_context"
    )
    recall_set = set(recall_ids)

    db_path = repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name / "rems_sim.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print(f"=== 回忆块前 {args.top} 条 ===")
    for i, eid in enumerate(recall_ids[: args.top], 1):
        row = conn.execute(
            "SELECT event_id, summaries, content_raw FROM events WHERE event_id=?",
            (eid,),
        ).fetchone()
        if not row:
            print(f"{i}. {eid} （库中无此事件）")
            continue
        summ = json.loads(row["summaries"] or "{}")
        text = summ.get("L2") or summ.get("l2") or summ.get("L1") or summ.get("l1") or ""
        raw_snip = (row["content_raw"] or "")[:70].replace("\n", " ")
        print(f"{i}. [{eid}] {text[:120]}")
        print(f"    raw: {raw_snip}...")
        print()

    keywords = ("龙禁尉", "戴权", "一千二百", "履历", "捐")
    print("=== 库内关键词相关事件（是否在回忆块）===")
    for row in conn.execute(
        "SELECT event_id, summaries, content_raw FROM events WHERE status='active'"
    ):
        blob = (row["content_raw"] or "") + json.dumps(
            row["summaries"] or "{}", ensure_ascii=False
        )
        if not any(k in blob for k in keywords):
            continue
        summ = json.loads(row["summaries"] or "{}")
        l2 = summ.get("L2") or summ.get("l2") or summ.get("L1") or summ.get("l1") or ""
        eid = row["event_id"]
        tag = "命中" if eid in recall_set else "未中"
        print(f"[{tag}] {eid}: {l2[:100]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
