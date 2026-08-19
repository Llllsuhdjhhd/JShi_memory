#!/usr/bin/env python3
"""Offline compare tri_band vs hybrid_literary on one chunk's act_query (no LLM).

Usage::

    python tests/scenarios/hongloumeng/compare_recall_profiles.py 77
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rems.config import REMSConfig, StorageConfig
from rems.observability.console_io import configure_stdio_utf8
from rems.pipeline import REMSPipeline
from rems.retrieval import apply_profile_defaults

configure_stdio_utf8()


def _load_act_query(debug_dir: Path) -> str:
    trace = json.loads((debug_dir / "recall_trace.json").read_text(encoding="utf-8"))
    entries = trace.get("query_act", {}).get("entries") or []
    if not entries:
        raise RuntimeError("no act_query in recall_trace")
    return entries[-1].get("act_text") or ""


def _rank_block(pipeline: REMSPipeline, act: str, profile: str) -> list[str]:
    cfg = pipeline.config
    cfg.recall_profile = profile
    apply_profile_defaults(cfg, profile)
    pipeline.recall_service.refresh_backend("dialogue")
    meta = pipeline.metabolism_service._repo.get_shadow()  # noqa: SLF001
    block = pipeline.recall_service.build_recall_block(
        act,
        meta,
        act_source=act,
        act_source_label="audit",
    )
    return [it.event_id for it in (block.items or [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_id", type=int)
    parser.add_argument("--run-name", default="continuous_run")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    repo = _REPO
    ws_db = (
        repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name / "rems_sim.db"
    )
    debug_dir = (
        repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name
        / "debug" / f"chunk_{args.chunk_id}"
    )
    if not ws_db.is_file():
        print(f"缺少数据库: {ws_db}", file=sys.stderr)
        return 1

    act = _load_act_query(debug_dir)
    cfg = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{ws_db.as_posix()}",
            qdrant_path=str(
                repo / "tests" / "scenarios" / "hongloumeng" / "outputs"
                / args.run_name / "qdrant_sim"
            ),
        ),
    )
    pipeline = REMSPipeline.from_config(cfg)

    tri = _rank_block(pipeline, act, "tri_band")[: args.top]
    hybrid = _rank_block(pipeline, act, "hybrid_literary")[: args.top]

    print(f"=== Chunk {args.chunk_id} act_query ({len(act)} chars) ===")
    print(act[:300])
    print()
    print("=== tri_band Top ===")
    for i, eid in enumerate(tri, 1):
        print(f"  {i:2d}. {eid}")
    print("=== hybrid_literary Top ===")
    for i, eid in enumerate(hybrid, 1):
        print(f"  {i:2d}. {eid}")
    only_tri = set(tri) - set(hybrid)
    only_hyb = set(hybrid) - set(tri)
    print(f"\n仅在 tri_band: {len(only_tri)}  仅在 hybrid: {len(only_hyb)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
