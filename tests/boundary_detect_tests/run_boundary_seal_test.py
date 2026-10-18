#!/usr/bin/env python3
# -*- coding: utf-8
"""苹果发布会语料：边界提示词、模型返回与封存结果评估。

在 ``tests/boundary_detect_tests/`` 下运行，默认 **离线**（FakeLLM + 手工边界 JSON）。
配置 ``REMS_LLM__API_KEY`` 后加 ``--live`` 可调用真实 ``boundary_detection`` / ``event_enrichment``。

::

    python tests/boundary_detect_tests/run_boundary_seal_test.py
    python tests/boundary_detect_tests/run_boundary_seal_test.py --live
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_TESTS = _REPO / "tests"
for p in (_SRC, _TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rems.config import REMSConfig, StorageConfig  # noqa: E402
from rems.llm.prompts import BOUNDARY_SYSTEM, BOUNDARY_USER, build_user_mode_block  # noqa: E402
from rems.port import MemoryExperience  # noqa: E402
from rems.llm.provider import LLMProvider  # noqa: E402
from rems.services.event_service import EventService  # noqa: E402
from rems.services.metabolism_service import MetabolismService  # noqa: E402
from rems.services.role_service import RoleService  # noqa: E402
from rems.skills.boundary_detection import BoundaryDetectionSkill  # noqa: E402
from rems.skills.event_enrichment import EventEnrichmentSkill  # noqa: E402
from rems.skills.role_extraction import RoleExtractionSkill  # noqa: E402
from rems.storage.database import Database  # noqa: E402
from rems.storage.repository import EventRepository, MetabolismRepository, RoleRepository  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "例子-苹果发布会.md"
OUT_BASE = Path(__file__).resolve().parent / "outputs"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_turns(md_path: Path) -> list[tuple[int, str]]:
    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"^###\s*(\d+)\s*$", text, flags=re.MULTILINE)
    turns: list[tuple[int, str]] = []
    # blocks: [preamble, num, body, num, body, ...]
    i = 1
    while i + 1 < len(blocks):
        num = int(blocks[i])
        body = blocks[i + 1].strip()
        if body:
            turns.append((num, body))
        i += 2
    if turns:
        return turns
    body = text.strip()
    if not body:
        raise ValueError(f"{md_path} 是空文件")
    return [(1, body)]


def _enrichment_stub(text: str) -> dict[str, Any]:
    body = text
    start = text.find("## 事件原文")
    if start >= 0:
        body = text[start + len("## 事件原文"):]
        end = body.find("## 已知角色")
        if end >= 0:
            body = body[:end]
    preview = shorten(" ".join(body.split()), width=48, placeholder="…")
    return {
        "summaries": {"L1": preview or "（无摘要）"},
        "emotion": {"joy": 0.35, "surprise": 0.2},
        "objects": [],
        "roles": [],
    }


def _offline_boundary_plan(
    turn_no: int,
    *,
    n_sent: int,
    residual_ids: list[str],
) -> dict[str, Any]:
    """与语料设计一致的离线边界 JSON（新协议 events/residual/no_form）。"""
    all_idx = list(range(1, n_sent + 1))
    rid = residual_ids[0] if residual_ids else None

    if turn_no == 1:
        return {"events": [], "residual": [{"indices": all_idx, "continues": None}], "no_form": []}
    if turn_no == 2:
        return {"events": [{"indices": all_idx, "continues": rid}], "residual": [], "no_form": []}
    if turn_no == 3:
        return {"events": [], "residual": [{"indices": all_idx, "continues": None}], "no_form": []}
    if turn_no == 4:
        return {"events": [{"indices": all_idx, "continues": rid}], "residual": [], "no_form": []}
    if turn_no == 5:
        return {"events": [], "residual": [{"indices": all_idx, "continues": None}], "no_form": []}
    if turn_no == 6:
        return {"events": [{"indices": all_idx, "continues": rid}], "residual": [], "no_form": []}
    if turn_no == 7:
        if n_sent == 1:
            return {"events": [], "residual": [], "no_form": [1]}
        return {"events": [], "residual": [{"indices": all_idx, "continues": None}], "no_form": []}
    if turn_no == 8:
        return {"events": [{"indices": all_idx, "continues": None}], "residual": [], "no_form": []}
    return {"events": [], "residual": [{"indices": all_idx, "continues": None}], "no_form": []}


class RecordingLLM:
    """Wrap real LLM or FakeLLM queue; persist every complete() call."""

    def __init__(
        self,
        inner: Any,
        log_dir: Path,
        *,
        offline_planner: Callable[..., dict[str, Any]] | None = None,
    ):
        self._inner = inner
        self._log_dir = log_dir
        self._offline_planner = offline_planner
        self._counter = 0
        self._boundary_turn = 0
        self._meta_repo: MetabolismRepository | None = None
        self._last_ctx: dict[str, Any] | None = None
        self.log_paths: list[Path] = []

    def bind_metabolism(self, meta_repo: MetabolismRepository) -> None:
        self._meta_repo = meta_repo

    def set_boundary_context(self, ctx: dict[str, Any], turn_no: int) -> None:
        self._last_ctx = ctx
        self._boundary_turn = turn_no

    def _write_log(self, payload: dict[str, Any]) -> Path:
        self._counter += 1
        name = f"{self._counter:03d}_{payload['task_type']}.json"
        path = self._log_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log_paths.append(path)
        return path

    def complete(self, task_type: str, messages: list[dict[str, str]], **kwargs) -> str:
        t0 = time.perf_counter()
        if task_type == "boundary_detection" and self._offline_planner and self._last_ctx:
            ucs = self._meta_repo.get_unclosed_events() if self._meta_repo else []
            residual_ids = [
                ue.id for ue in ucs if (ue.formation_role or "residual") == "residual"
            ]
            n_sent = len(self._last_ctx.get("sentences") or [])
            data = self._offline_planner(
                self._boundary_turn,
                n_sent=n_sent,
                residual_ids=residual_ids,
            )
            raw = json.dumps(data, ensure_ascii=False)
        elif self._offline_planner and task_type == "event_enrichment":
            user = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
            raw = json.dumps(_enrichment_stub(user), ensure_ascii=False)
        elif self._offline_planner and task_type == "summary":
            raw = "（离线占位）装饰句。"
        elif self._inner is None:
            raise RuntimeError(f"离线模式未覆盖的模型任务: {task_type}")
        else:
            raw = self._inner.complete(task_type, messages, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        self._write_log({
            "task_type": task_type,
            "latency_ms": ms,
            "messages": messages,
            "response": raw,
        })
        return raw

    def complete_json(self, task_type: str, messages: list[dict], **kwargs) -> dict[str, Any]:
        raw = self.complete(task_type, messages, **kwargs)
        if not isinstance(raw, str):
            return raw
        extract = getattr(self._inner, "_extract_json", None)
        if extract is not None:
            return extract(raw)
        return json.loads(raw)


def build_boundary_messages(skill: BoundaryDetectionSkill, config: REMSConfig, ctx: dict[str, Any]) -> list[dict[str, str]]:
    factor = float(getattr(config, "event_len_factor", 2) or 2)
    user_msg = BOUNDARY_USER.format(
        memory_buffer=ctx.get("indexed_input") or "（无）",
        ev_len=int(config.event_min_chars or 0),
        k=f"{factor:g}",
    )
    system_msg = build_user_mode_block(config) + BOUNDARY_SYSTEM
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _config_for_run(
    out_dir: Path,
    *,
    live: bool,
    event_min_chars: int,
    event_len_factor: float,
) -> REMSConfig:
    cfg = REMSConfig()
    cfg.storage = StorageConfig(
        database_url=f"sqlite:///{(out_dir / 'run.db').as_posix()}",
        qdrant_path=str(out_dir / "qdrant"),
    )
    cfg.embedding.provider = "hash"
    # 生产默认约 900 字才立即封存。本语料每轮很短，调试封存管线时用 0。
    cfg.event_min_chars = event_min_chars
    cfg.event_len_factor = event_len_factor
    cfg.ingest_llm_session_enabled = False
    if not live:
        cfg.llm.api_key = "offline"
    return cfg


def _event_md(ev) -> str:
    l1 = (ev.summaries or {}).get("L1", "")
    return (
        f"| `{ev.event_id}` | `{ev.seal_reason}` | {len(ev.content_raw)} | "
        f"{shorten(ev.content_raw, width=120, placeholder='…')} | {shorten(l1, width=60, placeholder='…')} |\n"
    )


def write_report(
    report_path: Path,
    *,
    turns: list[tuple[int, str]],
    turn_results: list[dict[str, Any]],
    llm_dir: Path,
    live: bool,
) -> None:
    lines: list[str] = [
        "# 苹果发布会 — 事件封存评估报告\n\n",
        f"- **生成时间**: {_iso_now()}\n",
        f"- **模式**: {'真实 LLM' if live else '离线 FakeLLM（边界 JSON 按语料设计脚本生成）'}\n",
        f"- **语料**: `例子-苹果发布会.md`（{len(turns)} 轮）\n",
        f"- **ev_len**: {turn_results[0]['min_chars'] if turn_results else '?'} 字；"
        f"**k**: {turn_results[0].get('factor', 2):g}\n",
        f"- **LLM 轨迹目录**: `{llm_dir.name}/`\n\n",
        "## 1. 测试目的\n\n",
        (
            "把整段客厅对话一次送入边界检测，看模型如何划分事件、残影和未形成句，以及程序据此封存了哪些事件。\n\n"
            if len(turns) == 1
            else
            "验证多轮叙述下边界划分与封存：闭环段落封存，未说完的留在残影，没有经验结构的句子不单独成事件。\n\n"
        ),
        "## 2. 轮次结果\n\n",
        "| 轮次 | 输入字数 | 当轮封存数 | 边界调用后 UC 数 | 残影字数 |\n",
        "|------|---------|-----------|-----------------|--------|\n",
    ]
    total_sealed = 0
    for row in turn_results:
        total_sealed += row["sealed_n"]
        lines.append(
            f"| {row['turn']} | {row['input_chars']} | {row['sealed_n']} | "
            f"{row['uc_n']} | {row['shadow_chars']} |\n"
        )
    lines.append(f"\n**累计封存事件**: {total_sealed}\n\n")

    lines.append("## 3. 封存事件明细\n\n")
    lines.append("| event_id | seal_reason | L0 字数 | content_raw（截断） | L1（截断） |\n")
    lines.append("|----------|-------------|--------|---------------------|------------|\n")
    for row in turn_results:
        for ev in row["sealed_events"]:
            lines.append(_event_md(ev))

    final = turn_results[-1] if turn_results else None
    lines.append("\n## 4. 终态（最后一轮后）\n\n")
    if final:
        lines.append("### 未完成事件 (UC)\n\n")
        if not final["uc_brief"]:
            lines.append("_（无）_\n\n")
        else:
            lines.append(final["uc_brief"] + "\n")
        lines.append("### 残影 (Shadow)\n\n")
        lines.append(f"{final['shadow_preview']}\n\n")

    lines.append("## 5. 提示词与模型返回\n\n")
    lines.append(
        "边界检测使用 `BOUNDARY_SYSTEM` + `BOUNDARY_USER`（见 `src/rems/llm/prompts.py`）。"
        "每轮实际填充内容与生成的 JSON 见下列文件：\n\n"
    )
    for p in sorted(llm_dir.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        task = data.get("task_type", "?")
        lines.append(f"- **`{p.name}`** · task=`{task}` · latency={data.get('latency_ms', 0):.0f}ms\n")
    lines.append("\n### 5.1 系统提示词（固定）\n\n```text\n")
    lines.append(BOUNDARY_SYSTEM.strip() + "\n```\n\n")

    boundary_logs = sorted(llm_dir.glob("*boundary_detection*.json"))
    for i, p in enumerate(boundary_logs, 1):
        data = json.loads(p.read_text(encoding="utf-8"))
        user_msg = next(
            (m["content"] for m in data.get("messages", []) if m.get("role") == "user"),
            "",
        )
        lines.append(f"### 5.2 第 {i} 次边界调用 — 用户提示词\n\n```text\n{user_msg}\n```\n\n")
        lines.append("### 5.3 模型返回（JSON）\n\n```json\n")
        resp = data.get("response", "")
        try:
            pretty = json.dumps(json.loads(resp), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pretty = resp
        lines.append(pretty + "\n```\n\n")

    lines.append("## 6. 评估结论\n\n")
    if len(turns) == 1:
        lines.extend(_evaluation_single_narrative(turn_results, llm_dir, live=live))
    else:
        lines.extend(_evaluation_section(turn_results, live=live))
    report_path.write_text("".join(lines), encoding="utf-8")


def _evaluation_single_narrative(
    turn_results: list[dict[str, Any]],
    llm_dir: Path,
    *,
    live: bool,
) -> list[str]:
    row = turn_results[-1] if turn_results else None
    sealed_n = row["sealed_n"] if row else 0
    events = row["sealed_events"] if row else []
    lengths = [len(ev.content_raw) for ev in events]
    boundary_logs = sorted(llm_dir.glob("*boundary_detection.json"))
    n_events = n_residual = n_no_form = 0
    if boundary_logs:
        data = json.loads(boundary_logs[-1].read_text(encoding="utf-8"))
        try:
            payload = json.loads(data.get("response") or "{}")
        except json.JSONDecodeError:
            payload = {}
        n_events = len(payload.get("events") or [])
        n_residual = len(payload.get("residual") or [])
        n_no_form = len(payload.get("no_form") or [])
    lines = [
        "语料是一整段客厅看发布会的对话，一次送入边界检测。\n\n",
        f"- 模型划出事件 {n_events} 条，残影 {n_residual} 条，未形成句子 {n_no_form} 句。\n",
        f"- 程序实际封存 {sealed_n} 条。\n",
    ]
    if lengths:
        lines.append(
            f"- 封存原文长度：最短 {min(lengths)} 字，最长 {max(lengths)} 字，"
            f"平均 {sum(lengths) / len(lengths):.0f} 字。\n"
        )
    if row and row["uc_n"]:
        lines.append(f"- 结束后仍有未完成事件 {row['uc_n']} 条，残影 {row['shadow_chars']} 字。\n")
    else:
        lines.append("- 结束后没有未完成事件，残影为空。\n")
    reasons: dict[str, int] = {}
    for ev in events:
        reasons[ev.seal_reason or ""] = reasons.get(ev.seal_reason or "", 0) + 1
    if reasons:
        lines.append(
            "- 封存原因："
            + "，".join(f"`{k}` {v} 条" for k, v in reasons.items())
            + "。\n"
        )
    if not live:
        lines.append("\n本次没有调用真实模型。\n")
    return lines


def _evaluation_section(turn_results: list[dict[str, Any]], *, live: bool) -> list[str]:
    sealed_total = sum(r["sealed_n"] for r in turn_results)
    turn2 = next((r for r in turn_results if r["turn"] == 2), None)
    turn4 = next((r for r in turn_results if r["turn"] == 4), None)
    turn6 = next((r for r in turn_results if r["turn"] == 6), None)
    turn7 = next((r for r in turn_results if r["turn"] == 7), None)
    turn8 = next((r for r in turn_results if r["turn"] == 8), None)

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "Vision Pro 段在轮次 2 产生封存",
        bool(turn2 and turn2["sealed_n"] >= 1),
        "预期：开场观看 + Vision Pro 宣布合并为一条 closed 事件。",
    ))
    checks.append((
        "MacBook Neo 在轮次 4 续写轮次 3 后封存",
        bool(turn4 and turn4["sealed_n"] >= 1),
        "预期：轮次 3 仅 residual，轮次 4 continues 同一 UC 后封存。",
    ))
    checks.append((
        "Apple Watch Ultra 3 在轮次 6 封存",
        bool(turn6 and turn6["sealed_n"] >= 1),
        "预期：轮次 5 截断句留在 residual，轮次 6 闭环。",
    ))
    checks.append((
        "轮次 7 单独「好的。」不触发封存",
        bool(turn7 and turn7["sealed_n"] == 0),
        "预期：no_form，不凭空生成事件。",
    ))
    checks.append((
        "轮次 8 个人收尾动作封存",
        bool(turn8 and turn8["sealed_n"] >= 1),
        "预期：关电视、热饭与发布会内容分离。",
    ))
    checks.append((
        "全流程至少 4 条基本事件",
        sealed_total >= 4,
        f"实际封存 {sealed_total} 条。",
    ))

    passed = sum(1 for _, ok, _ in checks if ok)
    lines = [
        f"- **自动检查**: {passed}/{len(checks)} 项通过\n\n",
        "| 检查项 | 结果 | 说明 |\n|--------|------|------|\n",
    ]
    for name, ok, note in checks:
        lines.append(f"| {name} | {'通过' if ok else '未通过'} | {note} |\n")

    if not live:
        lines.append(
            "\n> **说明**：当前为离线模式，边界 JSON 由 `run_boundary_seal_test.py` 内"
            " `_offline_boundary_plan` 生成，用于验证 **封存管线** 与提示词拼装；"
            "若要评估真实模型划分质量，请配置 `REMS_LLM__API_KEY` 后使用 `--live` 重跑。\n"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="苹果发布会边界 + 封存测试")
    parser.add_argument("--live", action="store_true", help="使用真实 LLM（需 API Key）")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=None, help="输出目录（默认 outputs/run_<ts>）")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="事件工作尺度 ev_len。0=不设尺度",
    )
    parser.add_argument("--factor", type=float, default=2.0, help="长度上限倍数 k")
    args = parser.parse_args()

    if args.live:
        cfg_probe = REMSConfig()
        if not (cfg_probe.llm.api_key or "").strip():
            print("缺少 REMS_LLM__API_KEY，无法 --live。", file=sys.stderr)
            return 2

    turns = parse_turns(args.corpus)
    run_dir = args.out or (OUT_BASE / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    llm_dir = run_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)

    cfg = _config_for_run(
        run_dir,
        live=args.live,
        event_min_chars=args.min_chars,
        event_len_factor=args.factor,
    )
    db = Database(cfg.storage.database_url)
    db.create_tables()
    meta_repo = MetabolismRepository(db)
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)

    inner = LLMProvider(cfg) if args.live else None
    recorder = RecordingLLM(
        inner,
        llm_dir,
        offline_planner=None if args.live else _offline_boundary_plan,
    )

    boundary = BoundaryDetectionSkill(recorder, cfg)
    role_skill = RoleExtractionSkill(recorder, cfg)
    enrich = EventEnrichmentSkill(recorder, cfg, role_fallback=role_skill)
    role_service = RoleService(cfg, role_repo, role_skill, llm=recorder)
    event_svc = EventService(
        cfg, recorder, event_repo, None, enrich, role_service=role_service,
    )
    metabolism = MetabolismService(cfg, meta_repo, boundary, event_svc, event_repo=event_repo)
    recorder.bind_metabolism(meta_repo)

    turn_results: list[dict[str, Any]] = []
    for turn_no, text in turns:
        shadow_before = meta_repo.get_shadow()
        ucs_before = meta_repo.get_unclosed_events()
        ctx = BoundaryDetectionSkill.build_index_context(
            shadow_before.content if shadow_before else "",
            text,
            ucs_before,
        )
        recorder.set_boundary_context(ctx, turn_no)
        # 预写一轮「仅边界」提示词快照，便于无 API 时审阅
        b_msgs = build_boundary_messages(boundary, cfg, ctx)
        (llm_dir / f"turn{turn_no:02d}_boundary_prompt_preview.json").write_text(
            json.dumps({"messages": b_msgs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        memory = MemoryExperience(
            subject_id="jshi-1",
            text=text,
            objects={"小米": "OBJ-XIAOMI", "小欢": "OBJ-XIAOHUAN"},
            source_ids=("apple-keynote",),
            segment_id="seg-apple",
        )
        sealed = metabolism.process_input(text, memory=memory)
        shadow_after = meta_repo.get_shadow()
        ucs_after = meta_repo.get_unclosed_events()
        turn_results.append({
            "turn": turn_no,
            "min_chars": cfg.event_min_chars,
            "factor": cfg.event_len_factor,
            "input_chars": len(text),
            "sealed_n": len(sealed),
            "sealed_events": sealed,
            "uc_n": len(ucs_after),
            "shadow_chars": len(shadow_after.content) if shadow_after else 0,
            "uc_brief": "\n".join(
                f"- `{ue.id}` · {shorten(ue.merged_content, width=200, placeholder='…')}"
                for ue in ucs_after
            ),
            "shadow_preview": shorten(
                (shadow_after.content if shadow_after else "") or "（空）",
                width=300,
                placeholder="…",
            ),
        })
        print(
            f"[turn {turn_no}] sealed={len(sealed)} uc={len(ucs_after)} "
            f"shadow={turn_results[-1]['shadow_chars']}ch",
            flush=True,
        )

    report_path = run_dir / "苹果发布会_封存评估报告.md"
    write_report(
        report_path,
        turns=turns,
        turn_results=turn_results,
        llm_dir=llm_dir,
        live=args.live,
    )
    print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
