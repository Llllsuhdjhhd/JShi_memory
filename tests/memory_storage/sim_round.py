#!/usr/bin/env python3
"""记忆质量模拟：多轮多对象对话 → 真实 ingest → 落位审计 → recall 验证。

用于"跳出审视"记忆与回忆质量：每轮模拟输入走真实 LLM 边界检测 + 事件充实，
落库后打印事件的 L0/L1/L2/L3、分级白描、8 维情绪、arousal/valence、遗忘因子，
再对指定查询跑 recall 看召回命中。

用法::

    python tests/memory_storage/sim_round.py --env C:/path/.env
    python tests/memory_storage/sim_round.py --rounds rounds_1.json
    python tests/memory_storage/sim_round.py --recall "杭州 阿明"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rems.llm.provider import LLMProvider
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience
from rems.storage.database import Database
from rems.storage.repository import EventRepository

_BASE = Path(__file__).resolve().parent
DB_NAME = "memory_storage_test.db"
QDRANT_DIR = "qdrant_test"
SUBJECT_DEFAULT = "jshi-1"

# 默认模拟：3 轮对话（每轮 匠石+deepseek 两段），覆盖事实/决策/情绪/多对象。
DEFAULT_ROUNDS = [
    {
        "turns": [
            {
                "speaker": "匠石",
                "content": "阿明约我下周四去杭州聚一聚，帮我查一下当天的高铁班次和天气，我准备坐早班车去。",
            },
            {
                "speaker": "deepseek",
                "content": "下周四杭州预报有中雨，建议你坐 7:05 的 G 字头早班车，9:20 左右到杭州东站，记得带伞，出站口到阿明约的茶室步行大概十分钟。",
            },
        ],
        "objects": {"deepseek": "OBJ-DEEPSEEK", "阿明": "OBJ-AMING"},
    },
    {
        "turns": [
            {
                "speaker": "匠石",
                "content": "今天下午的评审会上，陈姐把我提的模块重构方案否了，说我数据支撑不够，我心里挺不是滋味的。",
            },
            {
                "speaker": "deepseek",
                "content": "被否定确实不好受，但她的反馈里有一句值得抓住：补上基准测试和线上流量分布，方案的说服力会完全不同。你先休息一下，明天我们先把这两组数据拉出来。",
            },
        ],
        "objects": {"deepseek": "OBJ-DEEPSEEK", "陈姐": "OBJ-CHENJIE"},
    },
    {
        "turns": [
            {
                "speaker": "匠石",
                "content": "想起来上个月我和阿明在西湖边聊过这个项目，他还提过类似的想法。这次见面我把新笔记本带上，正好给他看现在的 demo。",
            },
            {
                "speaker": "deepseek",
                "content": "好主意，带设备当面演示比截图有说服力。建议提前一晚把 demo 跑通，别在茶室现场出岔子；另外可以顺手记下他对方案的意见，回来我们整理进下一版。",
            },
        ],
        "objects": {"deepseek": "OBJ-DEEPSEEK", "阿明": "OBJ-AMING"},
    },
]


def load_rounds(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise SystemExit(f"rounds 文件应为 JSON 数组: {path}")


def build_experiences(rounds: list[dict], subject_id: str) -> list[MemoryExperience]:
    exps: list[MemoryExperience] = []
    seq = 0
    for r in rounds:
        for t in r["turns"]:
            seq += 1
            exps.append(
                MemoryExperience(
                    subject_id=subject_id,
                    text=f"{t['speaker']}: {t['content']}",
                    objects=dict(r.get("objects") or {}),
                    source_ids=(f"sim-{datetime.now():%Y%m%d}-{seq:03d}",),
                    segment_id=f"sim-{seq:03d}",
                    origin="external",
                )
            )
    return exps


def audit(pipeline: REMSPipeline, result, experiences) -> None:
    print("\n" + "=" * 72)
    print("落位审计（sealed events）")
    print("=" * 72)
    for eid in result.sealed_event_ids:
        ev = pipeline.event_repo.get(eid)
        if ev is None:
            continue
        print(f"\n[{ev.event_id}] origin={ev.origin} act={ev.activation_energy:.3f} "
              f"ff={ev.forgetting_factor:.3f} 对象="
              f"{[r.role_id for r in ev.role_list if not r.is_subject]}")
        l0 = ev.content_raw.replace("\n", " | ")
        print(f"  L0: {l0[:160]}{'…' if len(l0) > 160 else ''}")
        for key in ("L1", "L2", "L3"):
            if key in (ev.summaries or {}):
                print(f"  {key}: {(ev.summaries[key] or '')[:200]}")
        for re_ in ev.role_list:
            if re_.is_subject:
                continue
            snap = re_.role_snapshot
            if snap is not None:
                print(f"  白描[{re_.role_id}] l1={snap.l1_mention or ''} | "
                      f"l2={snap.l2_interaction or ''} | l3={snap.l3_decision or ''}")
        if ev.emotion is not None:
            emo = ev.emotion
            print(
                f"  情绪: anger={emo.emotion.anger:.2f} fear={emo.emotion.fear:.2f} "
                f"joy={emo.emotion.joy:.2f} sadness={emo.emotion.sadness:.2f} "
                f"surprise={emo.emotion.surprise:.2f} disgust={emo.emotion.disgust:.2f} "
                f"trust={emo.emotion.trust:.2f} anticipation={emo.emotion.anticipation:.2f} "
                f"| arousal={emo.arousal:.2f} valence={emo.valence:.2f}"
            )
    print("\nstored_marks 归属:")
    for exp in experiences:
        ids = result.stored_marks.get(exp.segment_id, [])
        print(f"  {exp.segment_id} ({exp.text[:24]}…) -> {[i[:12] for i in ids]}")
    if result.errors:
        print("\n错误:", *result.errors, sep="\n  ")


def recall_check(pipeline: REMSPipeline, queries: list[str], subject_id: str) -> None:
    print("\n" + "=" * 72)
    print("回忆验证")
    print("=" * 72)
    for q in queries:
        frags = pipeline.recall_pipeline.recall(
            subject_id, q, limit=3, reinforce=False
        )
        print(f"\nQ: {q}")
        for f in frags:
            print(f"  {f.event_id[:12]} [{f.score:.4f}] {f.summary_level} | "
                  f"{(f.content or '')[:80].replace(chr(10), ' / ')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Memory quality simulation: multi-object rounds -> ingest -> audit -> recall",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--rounds", default=None, help="模拟轮次 JSON（默认内置 3 轮）")
    parser.add_argument("--db", default=str(_BASE / DB_NAME))
    parser.add_argument("--qdrant-path", default=str(_BASE / QDRANT_DIR))
    parser.add_argument("--subject", default=SUBJECT_DEFAULT)
    parser.add_argument("--env", default=None, help="加载 API key 环境变量文件")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="不 ingest，只审计现有事件并跑回忆验证（复用已落库数据）",
    )
    parser.add_argument("--recall", action="append", default=[], help="回忆验证查询（可多次）")
    args = parser.parse_args(argv)

    if args.env:
        env_path = Path(args.env).resolve()
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    import os

                    os.environ.setdefault(k.strip(), v.strip())

    rounds = (
        load_rounds(Path(args.rounds))
        if args.rounds
        else [dict(r) for r in DEFAULT_ROUNDS]
    )

    from rems.config import REMSConfig, StorageConfig

    db_path = Path(args.db).resolve()
    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{db_path.as_posix()}",
            qdrant_path=str(Path(args.qdrant_path).resolve()),
        )
    )
    # 与 run_storage_test 默认一致：hash 嵌入（对应 qdrant_test 的 64 维索引）。
    config.embedding.provider = "hash"
    if not config.llm.api_key:
        raise SystemExit("未找到 LLM API key（--env 指向 .env 或仓库根目录放 .env）")

    pipeline = REMSPipeline.from_config(config)
    if args.audit_only:
        result = type(
            "AuditResult",
            (),
            {
                "sealed_event_ids": [
                    ev.event_id
                    for ev in pipeline.event_repo.list_all(exclude_tombstoned=True)
                    if ev.subject_id == args.subject
                ],
                "stored_marks": {},
                "errors": [],
            },
        )()
        audit(pipeline, result, [])
    else:
        experiences = build_experiences(rounds, args.subject)
        print(f"模拟输入: {len(experiences)} 段 / {sum(len(e.text) for e in experiences)} 字")
        batch = MemoryBatch(subject_id=args.subject, experiences=tuple(experiences))
        result = pipeline.ingest_batch(batch)
        audit(pipeline, result, experiences)
    if args.recall:
        recall_check(pipeline, args.recall, args.subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
