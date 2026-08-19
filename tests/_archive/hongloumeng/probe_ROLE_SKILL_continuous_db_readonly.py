"""probe_ROLE_SKILL_continuous_db_readonly

只读探测 ``RoleExtractionSkill``：从 ``continuous_run/rems_sim.db`` 读出残影，
与当前输入拼成 ``combined_text``（与 ``REMSPipeline.ingest`` 预召回前一致），
调用真实 LLM，**不向数据库写入任何内容**。

打印：system / user 提示词、原始 JSON 返回、以及简短结构化分析。

用法示例::

    conda run -n py3125 python tests/scenarios/hongloumeng/probe_ROLE_SKILL_continuous_db_readonly.py \\
        --chunk 3

    conda run -n py3125 python tests/scenarios/hongloumeng/probe_ROLE_SKILL_continuous_db_readonly.py \\
        --text "他又叹气道……"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# 可从仓库根或 scenarios/hongloumeng 下运行
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rems.config import REMSConfig, UserMode  # noqa: E402
from rems.llm.provider import LLMProvider  # noqa: E402
from rems.skills.role_extraction import RoleExtractionSkill, RoleExtractionResult  # noqa: E402


def _load_shadow_readonly(db_path: Path) -> str:
    """SQLite URI 只读打开；仅 SELECT shadow.content。"""
    uri_path = db_path.resolve().as_posix()
    uri = f"file:{uri_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        row = conn.execute("SELECT content FROM shadow LIMIT 1").fetchone()
        if not row:
            return ""
        return row[0] or ""
    finally:
        conn.close()


def _load_dataset_chunk(repo_root: Path, index_1based: int) -> str:
    path = repo_root / "data" / "hongloumeng_dataset.json"
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    if index_1based < 1 or index_1based > len(chunks):
        raise SystemExit(f"chunk index {index_1based} out of range 1..{len(chunks)}")
    return chunks[index_1based - 1]["content"]


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n {title}\n{line}", flush=True)


class _TracingLLM:
    """把 ``RoleExtractionSkill`` 实际发出的 messages 原样打印后再转发给真实 Provider。"""

    def __init__(self, inner: LLMProvider):
        self._inner = inner

    def complete_json(self, task_type: str, messages: list[dict[str, str]], **kwargs: object) -> dict:
        _banner("LLM 请求：messages（与 Skill 传入顺序一致）")
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            print(f"\n--- message[{i}] role={role!r} ---", flush=True)
            print(content, flush=True)
        _banner(f"调用 {task_type!r}（真实 API）…")
        data = self._inner.complete_json(task_type, messages, **kwargs)
        _banner("模型返回：parse 前的 dict（complete_json 抽取结果）")
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        return data


def _analyze(combined_text: str, shadow_len: int, raw_len: int, result: RoleExtractionResult) -> None:
    _banner("分析（本地规则，非 LLM）")
    roles = result.roles
    print(
        f"- combined_text：总长 {len(combined_text)}（残影 {shadow_len} + 换行 + 本轮输入 {raw_len}）\n"
        f"- 抽到角色数：{len(roles)}",
        flush=True,
    )
    if not roles:
        print("  → 本轮模型未给出任何 role；Recall 侧焦点人物可能仅靠规则抽取。", flush=True)
        return
    imp_hist: dict[str, int] = {}
    filled_snap = 0
    for r in roles:
        imp_hist[r.importance] = imp_hist.get(r.importance, 0) + 1
        snap = r.snapshot
        if any([snap.l1_mention, snap.l2_interaction, snap.l3_decision]):
            filled_snap += 1
        emo = r.emotional_model
        nz = 0
        if emo is not None:
            d = emo.emotion.model_dump()
            nz = sum(1 for v in d.values() if isinstance(v, (int, float)) and abs(float(v)) > 1e-6)
        rid = r.role_id or "(模型未给 role_id)"
        print(
            f"  · {r.name!r} | id_hint={rid!r} | importance={r.importance!r} | "
            f"snapshot非空={'是' if any([snap.l1_mention, snap.l2_interaction, snap.l3_decision]) else '否'} | "
            f"情绪维非零≈{nz}",
            flush=True,
        )
    print(f"- importance 分布：{imp_hist}", flush=True)
    print(f"- 至少填了一档 snapshot 的角色：{filled_snap}/{len(roles)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只读 DB + 仅测 RoleExtractionSkill；打印 prompts 与 JSON；不写库。",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent / "outputs" / "continuous_run" / "rems_sim.db",
        help="continuous_run 的 SQLite 路径（只读打开）",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--chunk",
        type=int,
        metavar="N",
        help="使用 data/hongloumeng_dataset.json 中第 N 块（从 1 起）的 content 作为本轮输入",
    )
    g.add_argument("--text", type=str, help="本轮原始输入文本（与 pipeline raw_input 一致）")
    args = parser.parse_args()

    db_path: Path = args.db
    if not db_path.is_file():
        raise SystemExit(f"DB 不存在: {db_path}（先跑 continuous_simulation 生成）")

    shadow = _load_shadow_readonly(db_path)
    if args.chunk is not None:
        raw = _load_dataset_chunk(_REPO_ROOT, args.chunk)
    else:
        raw = args.text or ""

    combined = (shadow + "\n" + raw).strip()

    _banner("输入摘要（只读 DB，不写）")
    print(
        f"DB (ro): {db_path}\n"
        f"残影长度: {len(shadow)}\n"
        f"本轮输入长度: {len(raw)}\n"
        f"combined_text 长度: {len(combined)}",
        flush=True,
    )

    config = REMSConfig()
    config.user_mode = UserMode.MULTI
    llm = LLMProvider(config)
    skill = RoleExtractionSkill(_TracingLLM(llm), config)

    result = skill.extract(combined, known_roles=None, budget=None)
    _banner("Skill 解析后的 RoleExtractionResult（Pydantic → JSON）")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), flush=True)
    _analyze(combined, len(shadow), len(raw), result)


if __name__ == "__main__":
    main()
