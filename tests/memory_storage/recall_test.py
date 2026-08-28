#!/usr/bin/env python3
"""回忆（Recall）测试程序：真实查询标注集 → Recall@K / MRR（P2 测量闭环）。

读取累积测试库（不重置），对每条标注查询跑 ``RecallPipeline.recall``，
输出 Recall@1/3/5、MRR、每查询命中明细，支持通道消融与只读校验。

用法::

    python tests/memory_storage/recall_test.py
    python tests/memory_storage/recall_test.py --channels semantic,lexical,object
    python tests/memory_storage/recall_test.py --no-reinforce
    python tests/memory_storage/recall_test.py --query "新开分支"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rems.recall import (
    HashEmbedding,
    NullReranker,
    QdrantRecallVectorStore,
    RecallPipeline,
    SentenceTransformerEmbedding,
)
from rems.recall.intent import RuleIntentClassifier
from rems.storage.database import Database
from rems.storage.repository import EventRepository

_BASE = Path(__file__).resolve().parent
DB_NAME = "memory_storage_test.db"
QDRANT_DIR = "qdrant_test"
QUERIES_FILE = "recall_queries.jsonl"
SUBJECT_DEFAULT = "jshi-1"
KNOWN_OBJECTS_DEFAULT = {"deepseek": "OBJ-DEEPSEEK"}


def load_queries(path: Path) -> list[dict]:
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pipeline(args, known_objects: dict[str, str]) -> RecallPipeline:
    db = Database(f"sqlite:///{Path(args.db).resolve().as_posix()}")
    repo = EventRepository(db)
    if args.embedding == "local":
        embedding = SentenceTransformerEmbedding(args.model or "BAAI/bge-base-zh-v1.5")
    else:
        embedding = HashEmbedding()
    vector_store = QdrantRecallVectorStore(args.collection, path=args.qdrant_path)
    classifier = RuleIntentClassifier(
        entity_lexicon=args.entity_lexicon or None,
        emotion_lexicon=args.emotion_lexicon or None,
        fact_lexicon=args.fact_lexicon or None,
        known_objects=known_objects,
    )
    return RecallPipeline(
        embedding,
        vector_store,
        repo,
        None,
        reranker=NullReranker(),
        intent_classifier=classifier,
        recency_enabled=args.recency,
        recency_window_events=args.recency_window,
        recency_top_k=args.recency_top_k,
    )


def snapshot_forgetting(db_path: str) -> dict[str, float]:
    db = Database(f"sqlite:///{Path(db_path).resolve().as_posix()}")
    repo = EventRepository(db)
    return {
        ev.event_id: float(getattr(ev, "forgetting_factor", 1.0) or 1.0)
        for ev in repo.list_all(exclude_tombstoned=True)
    }


def recall_at_k(hits: list[dict], expected: list[str]) -> tuple[dict[int, bool], float]:
    """Return (hit_at_k, reciprocal_rank)."""
    expected_set = set(expected)
    hit_at: dict[int, bool] = {k: False for k in (1, 3, 5)}
    rr = 0.0
    for rank, h in enumerate(hits, 1):
        if h["event_id"] in expected_set:
            if rr == 0.0:
                rr = 1.0 / rank
            for k in hit_at:
                if rank <= k:
                    hit_at[k] = True
    return hit_at, rr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recall evaluation on annotated queries (Recall@K / MRR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default=str(_BASE / DB_NAME))
    parser.add_argument("--qdrant-path", default=str(_BASE / QDRANT_DIR))
    parser.add_argument("--collection", default="rems_events")
    parser.add_argument("--queries", default=str(_BASE / QUERIES_FILE))
    parser.add_argument("--subject", default=SUBJECT_DEFAULT)
    parser.add_argument("--limit", type=int, default=5, help="每条查询返回 top N（默认 5）")
    parser.add_argument(
        "--channels",
        default=None,
        help="逗号分隔通道：semantic,lexical,object,recency,anchor（默认全部）",
    )
    parser.add_argument("--recency", action="store_true", help="启用近因通道")
    parser.add_argument("--recency-window", type=int, default=80)
    parser.add_argument("--recency-top-k", type=int, default=40)
    parser.add_argument(
        "--reinforce",
        action="store_true",
        help="启用强化写回（默认关闭：测量不污染记忆强度；开启用于行为验证）",
    )
    parser.add_argument("--embedding", choices=["hash", "local"], default="hash")
    parser.add_argument("--model", default=None, help="local 嵌入模型名")
    parser.add_argument("--query", default=None, help="只跑单条查询（不走标注集）")
    parser.add_argument("--known-object", action="append", default=[], metavar="NAME=ID")
    parser.add_argument("--entity-lexicon", action="append", default=[])
    parser.add_argument("--emotion-lexicon", action="append", default=[])
    parser.add_argument("--fact-lexicon", action="append", default=[])
    args = parser.parse_args(argv)

    known_objects: dict[str, str] = dict(KNOWN_OBJECTS_DEFAULT)
    for kv in args.known_object:
        if "=" in kv:
            name, oid = kv.split("=", 1)
            known_objects[name.strip()] = oid.strip()

    pipe = build_pipeline(args, known_objects)
    before = snapshot_forgetting(args.db) if not args.reinforce else None

    if args.query:
        queries = [{"query": args.query, "expected": [], "note": "single-query"}]
    else:
        queries = load_queries(Path(args.queries))
        if not queries:
            print(f"标注集为空: {args.queries}")
            return 1

    channels = tuple(c.strip() for c in args.channels.split(",")) if args.channels else None
    print(f"subject={args.subject}  channels={channels or 'all'}  reinforce={'on' if args.reinforce else 'off（只读）'}")
    print(f"{'query':<34} {'R@1':>4} {'R@3':>4} {'R@5':>4} {'MRR':>5}  top hits")

    agg = {"r@1": 0, "r@3": 0, "r@5": 0, "mrr": 0.0}
    for i, q in enumerate(queries, 1):
        frags = pipe.recall(
            args.subject,
            q["query"],
            limit=args.limit,
            channels=channels,
            reinforce=args.reinforce,
        )
        hits = [
            {
                "event_id": f.event_id,
                "score": f.score,
                "level": f.summary_level,
                "content": (f.content or "")[:60].replace("\n", " / "),
            }
            for f in frags
        ]
        hit_at, rr = recall_at_k(hits, q.get("expected") or [])
        agg["r@1"] += int(hit_at[1])
        agg["r@3"] += int(hit_at[3])
        agg["r@5"] += int(hit_at[5])
        agg["mrr"] += rr

        marked = "  ".join(
            f"#{j} {h['event_id'][:12]}[{h['score']:.4f}]{h['level']} {h['content']}"
            for j, h in enumerate(hits[:3], 1)
        )
        print(
            f"{q['query'][:32]:<34} {int(hit_at[1]):>4} {int(hit_at[3]):>4} {int(hit_at[5]):>4} {rr:>5.3f}  {marked}"
        )

    n = len(queries)
    print("-" * 100)
    print(
        f"mean Recall@1={agg['r@1']/n:.3f}  Recall@3={agg['r@3']/n:.3f}  "
        f"Recall@5={agg['r@5']/n:.3f}  MRR={agg['mrr']/n:.3f}  (n={n})"
    )

    if not args.reinforce:
        after = snapshot_forgetting(args.db)
        changed = {
            eid for eid in before if before[eid] != after.get(eid)
        }
        print(f"只读校验: forgetting_factor 变化 {len(changed)} 条 "
              f"({'通过' if not changed else '失败: ' + str(sorted(changed)[:3])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
