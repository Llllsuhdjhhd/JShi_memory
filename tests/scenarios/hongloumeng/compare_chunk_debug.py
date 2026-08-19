#!/usr/bin/env python3
"""Compare chunk debug bundles (baseline vs rerun)."""

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
from rems.ingest.focus_roles import match_focus_role_ids
from rems.storage.database import Database
from rems.storage.repository import RoleRepository

configure_stdio_utf8()

from tests.scenarios.common.harness import find_repo_root


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _step(pipe: dict, name: str) -> dict:
    for s in pipe.get("steps") or []:
        if s.get("step") == name:
            return s
    return {}


def _recall_ids(pipe: dict) -> list[str]:
    return list(_step(pipe, "recall_build_context").get("event_ids") or [])


def _summ(conn: sqlite3.Connection, eid: str) -> str:
    row = conn.execute("SELECT summaries FROM events WHERE event_id=?", (eid,)).fetchone()
    if not row:
        return "（库中无）"
    s = json.loads(row[0] or "{}")
    return (s.get("L2") or s.get("L1") or "")[:100]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=77)
    parser.add_argument("--run-name", default="continuous_run")
    parser.add_argument("--baseline-suffix", default="_baseline")
    args = parser.parse_args()

    repo = find_repo_root()
    root = repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name / "debug"
    new_dir = root / f"chunk_{args.chunk}"
    old_dir = root / f"chunk_{args.chunk}{args.baseline_suffix}"
    for d in (new_dir, old_dir):
        if not d.is_dir():
            print(f"缺少目录: {d}", file=sys.stderr)
            return 1

    new_pipe = _load(new_dir / "pipeline_trace.json")
    old_pipe = _load(old_dir / "pipeline_trace.json")
    new_report = _load(new_dir / "report.json")
    old_report = _load(old_dir / "report.json")

    db = repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name / "rems_sim.db"
    conn = sqlite3.connect(db)

    print(f"=== Chunk {args.chunk} 重跑对比 ===\n")

    def row(label: str, old_val, new_val):
        mark = " ← 变化" if old_val != new_val else ""
        print(f"{label}: {old_val} → {new_val}{mark}")

    row("events_total", old_report["events_total"], new_report["events_total"])
    row("events_created", old_report["events_created"], new_report["events_created"])
    row("elapsed_ms", f"{old_report['elapsed_ms']:.0f}", f"{new_report['elapsed_ms']:.0f}")

    old_pre = _step(old_pipe, "pre_recall_role_extract")
    new_pre = _step(new_pipe, "pre_recall_role_extract")
    row("focus 数量", len(old_pre.get("focus_role_ids") or []), len(new_pre.get("focus_role_ids") or []))

    db_roles = RoleRepository(Database(f"sqlite:///{db.as_posix()}"))
    id_to_name = {r.role_id: r.name for r in db_roles.list_all()}

    def fmt_focus(step: dict) -> str:
        ids = step.get("focus_role_ids") or []
        return ", ".join(id_to_name.get(i, i[:12]) for i in ids) or "（无）"

    print(f"\n--- focus_role_ids ---")
    print(f"  改前: {fmt_focus(old_pre)}")
    print(f"  改后: {fmt_focus(new_pre)}")

    old_act = _step(old_pipe, "recall_query_compress").get("act_chars")
    new_act = _step(new_pipe, "recall_query_compress").get("act_chars")
    row("\nact_query 字数", old_act, new_act)

    old_shadow = _step(old_pipe, "shadow_load").get("shadow_chars")
    new_shadow = _step(new_pipe, "shadow_load").get("shadow_chars")
    row("残影字数（recall 前）", old_shadow, new_shadow)

    old_ids = _recall_ids(old_pipe)
    new_ids = _recall_ids(new_pipe)
    old_set, new_set = set(old_ids), set(new_ids)
    overlap = old_set & new_set
    print(f"\n--- 回忆块 ---")
    print(f"  改前 Top 列表: {len(old_ids)} 条（trace 截断显示 {len(old_ids)}）")
    print(f"  改后 Top 列表: {len(new_ids)} 条")
    print(f"  交集: {len(overlap)}  仅改前: {len(old_set - new_set)}  仅改后: {len(new_set - old_set)}")

    keywords = {
        "协理/领牌": ("领牌", "对牌", "协理", "分派", "王兴"),
        "林如海/黛玉": ("林如海", "黛玉", "昭儿", "送灵", "苏州"),
        "秦钟/宝玉访凤姐": ("秦钟", "抱厦", "香灯"),
    }
    print("\n--- 关键词命中（回忆块内）---")
    for label, kws in keywords.items():
        def hits(ids: set[str]) -> list[str]:
            out = []
            for eid in ids:
                blob = _summ(conn, eid)
                if any(k in blob for k in kws):
                    out.append(eid)
            return out

        oh, nh = hits(old_set), hits(new_set)
        print(f"  [{label}] 改前 {len(oh)} / 改后 {len(nh)}")

    print("\n--- 改前 Top 10 ---")
    for i, eid in enumerate(old_ids[:10], 1):
        print(f"  {i}. {_summ(conn, eid)}")

    print("\n--- 改后 Top 10 ---")
    for i, eid in enumerate(new_ids[:10], 1):
        print(f"  {i}. {_summ(conn, eid)}")

    print("\n--- 仅改后新进 Top15 ---")
    for i, eid in enumerate([x for x in new_ids if x not in old_set][:15], 1):
        print(f"  + {i}. [{eid}] {_summ(conn, eid)}")

    print("\n--- 近因协理链（chunk72-76 封存，是否在回忆块）---")
    recent = conn.execute(
        "SELECT event_id, summaries FROM events ORDER BY rowid DESC LIMIT 20"
    ).fetchall()
    for eid, summ in recent:
        s = json.loads(summ or "{}")
        l2 = (s.get("L2") or s.get("L1") or "")[:70]
        o = "前" if eid in old_set else "  "
        n = "后" if eid in new_set else "  "
        print(f"  [{o}{n}] {eid}: {l2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
