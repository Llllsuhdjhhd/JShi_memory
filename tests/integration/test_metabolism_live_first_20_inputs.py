"""Live metabolism test: first 20 user inputs → basic events + shadow (残影).

This file is **not** a fast unit test. It calls your configured real LLM (``.env`` /
``REMS_LLM__*``) and, by default, **real** Chroma embeddings via
``SentenceTransformerEmbeddingFunction`` (``config.embedding.model_name``, usually
``BAAI/bge-small-zh-v1.5``), which requires a working **PyTorch** stack.

**Use the conda env where torch works** (your ``py3125`` is verified OK). If ``pytest``
runs with the **base** ``anaconda3\\python.exe``, torch may fail (WinError 1114 on
``c10.dll``); switch interpreter or run::

    conda activate py3125
    $env:REMS_RUN_LIVE_METABOLISM_TEST = "1"
    python -m pytest tests/integration/test_metabolism_live_first_20_inputs.py -v -s

Or without activating::

    conda run -n py3125 --no-capture-output python -m pytest tests/integration/test_metabolism_live_first_20_inputs.py -v -s

**Escape hatch** (no PyTorch; fake vectors only — not for semantic recall quality)::

    $env:REMS_LIVE_ALLOW_FAKE_EMBEDDING = "1"

After a successful run, open the generated Markdown report::

    tests/reports/metabolism_live_20_last_run.md

The report file is refreshed **after each input round** (and created right after the header)
so you can open it while pytest is still running.

Each remote LLM call prints **one short line** to the console (``[live-metabolism]``) —
this tracing lives **only in this test file** (wraps ``pipeline.llm.complete``), not in
``LLMProvider``.

Optional: ``$env:REMS_LIVE_METABOLISM_MAX_TURNS = "3"`` to run only the first N scripted inputs.

Scope (by design):
    Uses ``REMSPipeline`` wiring but calls ``metabolism_service.process_input`` only,
    so **ingest** extras (recall-based reconsolidation, per-seal ``check_and_abstract``,
    role white-painting / semantic-card refresh) are skipped to keep the scenario focused
    on **boundary detection → seal_event → shadow / unclosed** and to reduce duplicate
    LLM traffic. Basic events are still fully enriched inside ``EventService.seal_event``.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from unittest.mock import patch

import pytest

from rems.config import REMSConfig, StorageConfig
from rems.pipeline import REMSPipeline

class FakeEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[0.1] * 128 for _ in input]


def _fake_embedding_escape_enabled() -> bool:
    return os.environ.get("REMS_LIVE_ALLOW_FAKE_EMBEDDING", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _probe_torch() -> tuple[bool, str]:
    """Return (ok, human-readable detail including sys.executable)."""
    exe = sys.executable
    try:
        import torch

        _ = torch.zeros(1)
        return True, f"`torch` {torch.__version__} · 解释器: `{exe}`"
    except OSError as e:
        return False, f"解释器: `{exe}`\n`OSError`: {e!r}"
    except ImportError as e:
        return False, f"解释器: `{exe}`\n`ImportError`: {e!r}"

# 20 条递进式中文输入：模拟「碎片化叙述 → 逐步闭环」，便于观察残影累积与封存。
TWENTY_INPUTS: list[str] = [
    "早上八点我出门，天气有点阴。",
    "在小区门口买了一杯美式咖啡。",
    "咖啡有点酸，我不太喜欢酸味。",
    "坐地铁去公司，路上看了几封邮件。",
    "到公司后先开了站会，讨论了项目进度。",
    "站会结束后我继续做 A 项目的接口联调。",
    "中午和同事在楼下吃了盖饭。",
    "下午三点左右终于把联调问题修完了。",
    "然后写了一段简单的复盘笔记。",
    "傍晚离开公司前把桌面整理了一下。",
    "晚上回家路上顺便取了快递。",
    "回家后先洗了个澡。",
    "接着做了半小时拉伸。",
    "九点开始看了一会儿小说放松一下。",
    "十点半准备睡觉，设了明天的闹钟。",
    "躺下后又想起明天要带伞。",
    "最后确认了一下明早会议的材料在包里。",
    "就这样慢慢睡着了。",
    "（次日）早上闹钟响了，我按时起床。",
    "出门时确实带了伞，路上没下雨。",
]


def _live_input_turns() -> list[str]:
    """All scripted inputs, or first N when ``REMS_LIVE_METABOLISM_MAX_TURNS`` is set."""
    raw = os.environ.get("REMS_LIVE_METABOLISM_MAX_TURNS", "").strip()
    if not raw:
        return list(TWENTY_INPUTS)
    try:
        n = int(raw)
    except ValueError:
        return list(TWENTY_INPUTS)
    n = max(1, min(n, len(TWENTY_INPUTS)))
    return TWENTY_INPUTS[:n]


def _install_live_llm_console_trace(pipeline: REMSPipeline) -> None:
    """Wrap ``pipeline.llm.complete`` so each API round-trip prints one flushed line (test-only)."""
    llm = pipeline.llm
    _orig = llm.complete

    def _traced_complete(
        task_type: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        model = llm._get_model(task_type)
        pch = sum(len(m.get("content") or "") for m in messages)
        t0 = time.perf_counter()
        try:
            out = _orig(task_type, messages, temperature=temperature)
        except Exception as exc:
            print(
                f"[live-metabolism] LLM 失败 task={task_type} model={model} "
                f"prompt≈{pch}ch ({time.perf_counter() - t0:.1f}s) → {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        dt = time.perf_counter() - t0
        tail = (out or "").replace("\r", " ").replace("\n", " ").strip()
        if len(tail) > 72:
            tail = tail[:71] + "…"
        print(
            f"[live-metabolism] LLM task={task_type} model={model} prompt≈{pch}ch "
            f"reply≈{len(out or '')}ch ({dt:.1f}s) ↩ {tail}",
            flush=True,
        )
        return out

    llm.complete = _traced_complete  # type: ignore[method-assign]


def _write_live_report_md(
    report_path: Path,
    sections: list[str],
    *,
    progress_turn: int | None,
    total_turns: int,
) -> None:
    body = "".join(sections)
    if progress_turn is not None and progress_turn < total_turns:
        body += (
            "\n\n---\n\n"
            f"> **进行中**：已完成 **{progress_turn}** / {total_turns} 轮。"
            " 本文件每轮结束都会保存；全部完成后会追加全局快照与轮次表。\n"
        )
    report_path.write_text(body, encoding="utf-8")


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _live_metabolism_enabled() -> bool:
    return os.environ.get("REMS_RUN_LIVE_METABOLISM_TEST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _isolated_config(tmp_path: Path) -> REMSConfig:
    """Load ``.env`` via ``REMSConfig``, but force SQLite + Chroma under *tmp_path*."""
    base = REMSConfig()
    db_path = tmp_path / "live_metabolism.sqlite3"
    chroma_path = tmp_path / "chroma_live_metabolism"
    return base.model_copy(
        update={
            "storage": StorageConfig(
                database_url=f"sqlite:///{db_path.as_posix()}",
                qdrant_path=str(chroma_path),
            ),
        },
    )


def _md_escape_block(text: str, limit: int = 12000) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n\n…（截断，全文 {len(text)} 字符）…"
    return text


def _event_brief_md(ev) -> str:
    l1 = ev.summaries.get("L1", "") if ev.summaries else ""
    roles = ", ".join(r.role_id for r in ev.role_list) if ev.role_list else "（无）"
    return (
        f"- **{ev.event_id}** · 状态 `{ev.status.value}` · 角色: {roles}\n"
        f"  - **L0**（前 200 字）: {shorten(ev.content_raw, width=200, placeholder='…')}\n"
        f"  - **L1**（前 200 字）: {shorten(l1, width=200, placeholder='…')}\n"
    )


def _unclosed_brief_md(ue_list) -> str:
    if not ue_list:
        return "_（当前无未完成事件）_\n"
    lines: list[str] = []
    for ue in ue_list:
        lines.append(f"- **{ue.id}** · 片段数 {len(ue.content_fragments)} · 总长 {ue.total_length}\n")
        merged = shorten(ue.merged_content, width=300, placeholder="…")
        lines.append(f"  - 合并预览: {merged}\n")
    return "".join(lines)


@pytest.mark.live_llm
def test_first_20_inputs_metabolism_detailed_report(tmp_path: Path) -> None:
    """Drive 20 turns of ``process_input`` and write ``tests/reports/metabolism_live_20_last_run.md``."""
    if not _live_metabolism_enabled():
        pytest.skip(
            "Set environment variable REMS_RUN_LIVE_METABOLISM_TEST=1 to run this live LLM test."
        )

    cfg = _isolated_config(tmp_path)
    if not (cfg.llm.api_key or "").strip():
        pytest.skip("REMS_LLM__API_KEY is empty — configure .env before running this test.")

    embedding_note = ""
    if _fake_embedding_escape_enabled():
        with patch("rems.storage.vector_store.SentenceTransformerEmbeddingFunction") as mock_ef:
            mock_ef.return_value = FakeEmbeddingFunction()
            pipeline = REMSPipeline.from_config(cfg)
        embedding_note = (
            "- **Chroma 嵌入**: **假向量**（已设置 `REMS_LIVE_ALLOW_FAKE_EMBEDDING=1`），"
            "仅用于无 PyTorch 时的管线冒烟；语义检索无意义。\n"
        )
    else:
        torch_ok, torch_msg = _probe_torch()
        if not torch_ok:
            pytest.skip(
                "当前解释器无法加载 PyTorch，无法使用真实向量嵌入。\n"
                f"{torch_msg}\n\n"
                "请改用已安装可用 torch 的环境（例如）：`conda activate py3125` 后再运行 pytest，\n"
                "或使用：`conda run -n py3125 python -m pytest tests/integration/test_metabolism_live_first_20_inputs.py -v -s`\n\n"
                "若仅需在无 torch 的解释器上试跑管线，可设置 `REMS_LIVE_ALLOW_FAKE_EMBEDDING=1`（假嵌入）。"
            )
        pipeline = REMSPipeline.from_config(cfg)
        embedding_note = (
            f"- **Chroma 嵌入**: 真实本地模型 `{cfg.embedding.model_name}`（{torch_msg}）\n"
        )

    _install_live_llm_console_trace(pipeline)

    meta = pipeline.meta_repo
    events_repo = pipeline.event_repo

    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "metabolism_live_20_last_run.md"

    inputs = _live_input_turns()
    n_turns = len(inputs)

    sections: list[str] = []
    sections.append(f"# REMS 代谢实况：前 {n_turns} 次输入\n\n")
    sections.append(f"- 生成时间: {_iso_now()}\n")
    sections.append(
        f"- 配置摘要: `context_window={cfg.context_window}`, "
        f"`len_msg={cfg.len_msg}`, `physical_redline={cfg.physical_redline}`, "
        f"`safe_watermark={cfg.safe_watermark}`\n"
    )
    sections.append(
        f"- LLM: `base_url={cfg.llm.base_url}`（密钥未写入本文档）\n"
    )
    sections.append(embedding_note or "- **Chroma 嵌入**: （未记录）\n")
    sections.append(
        "\n## 测试范围说明\n\n"
        "本轮仅调用 **`MetabolismService.process_input`**（不经由 `REMSPipeline.ingest`），"
        "因此不会触发：回忆块组装、基于回忆的再巩固抽象、以及封存后的 `check_and_abstract` 向量聚类；"
        "也不会执行 **`RoleService.update_from_event`**（白描时间线 / 语义卡片 LLM 刷新）。\n\n"
        "仍会完整执行：**边界检测 → 对已闭环片段 `seal_event`（递归摘要、角色抽取、decoration）→ 更新残影与未完成库**。\n\n"
        "---\n\n"
    )

    _write_live_report_md(report_path, sections, progress_turn=0, total_turns=n_turns)
    print(
        "\n[live-metabolism] 报告已开始写入（每轮刷新）:\n"
        f"  {report_path.resolve()}\n"
        f"  本轮共 {n_turns} 条输入；控制台会打印每次 LLM 单行摘要（[live-metabolism]）。\n",
        flush=True,
    )

    turn_records: list[dict] = []
    turn_idx = 0

    try:
        for turn_idx, raw in enumerate(inputs, start=1):
            print(f"\n[live-metabolism] === 第 {turn_idx}/{n_turns} 轮 user 输入（{len(raw)} 字）===\n", flush=True)
            sealed = pipeline.metabolism_service.process_input(raw)
            shadow = meta.get_shadow()
            unclosed = meta.get_unclosed_events()

            turn_records.append(
                {
                    "turn": turn_idx,
                    "input": raw,
                    "sealed_count": len(sealed),
                    "sealed_ids": [e.event_id for e in sealed],
                    "shadow_len": shadow.length,
                    "unclosed_count": len(unclosed),
                }
            )

            sections.append(f"## 第 {turn_idx} 轮输入\n\n")
            sections.append("### 本轮用户原文\n\n")
            sections.append("```text\n")
            sections.append(_md_escape_block(raw, limit=4000))
            sections.append("\n```\n\n")

            sections.append("### 本轮新封存的基本事件\n\n")
            if not sealed:
                sections.append("_本轮边界模型未产出已闭环片段，故无新 `Event` 入库。_\n\n")
            else:
                for ev in sealed:
                    sections.append(_event_brief_md(ev))
                    sections.append("\n```json\n")
                    try:
                        sections.append(_md_escape_block(ev.model_dump_json(indent=2), limit=8000))
                    except Exception:
                        sections.append("（model_dump_json 失败，略）\n")
                    sections.append("\n```\n\n")

            sections.append("### 本轮结束后的残影（Shadow）\n\n")
            sections.append(f"- **字符数**: {shadow.length}\n")
            if shadow.updated_at:
                sections.append(f"- **updated_at**: {shadow.updated_at}\n")
            sections.append("\n```text\n")
            sections.append(_md_escape_block(shadow.content or ""))
            sections.append("\n```\n\n")

            sections.append("### 本轮结束后的未完成事件库（Unclosed）\n\n")
            sections.append(_unclosed_brief_md(unclosed))
            sections.append("\n---\n\n")

            _write_live_report_md(report_path, sections, progress_turn=turn_idx, total_turns=n_turns)
            print(
                f"[live-metabolism] 第 {turn_idx} 轮结束：新封存 {len(sealed)} 条，"
                f"残影 {shadow.length} 字，未完成 {len(unclosed)} 条\n",
                flush=True,
            )

    except Exception as e:
        sections.append(
            f"\n\n## 运行异常（约第 {turn_idx} 轮）\n\n```\n{type(e).__name__}: {e}\n```\n"
        )
        _write_live_report_md(report_path, sections, progress_turn=None, total_turns=n_turns)
        print(f"\n[live-metabolism] 已写入部分报告: {report_path.resolve()}\n", flush=True)
        raise

    all_basic = events_repo.list_all(is_abstract=False, exclude_tombstoned=True)
    sections.append(f"## {n_turns} 轮结束后的全局快照\n\n")
    sections.append(f"- **库中基本事件总数**（非抽象、非墓碑）: **{len(all_basic)}**\n\n")
    sections.append("### 按 `create_time` 排序的基本事件一览\n\n")
    for ev in all_basic:
        sections.append(_event_brief_md(ev))

    final_shadow = meta.get_shadow()
    final_unclosed = meta.get_unclosed_events()
    sections.append("\n### 最终残影\n\n")
    sections.append(f"- 字符数: **{final_shadow.length}**\n\n```text\n")
    sections.append(_md_escape_block(final_shadow.content or ""))
    sections.append("\n```\n\n### 最终未完成事件\n\n")
    sections.append(_unclosed_brief_md(final_unclosed))

    sections.append("\n## 轮次摘要表\n\n")
    sections.append("| 轮次 | 输入字数 | 新封存事件数 | 新事件 ID | 残影字数 | 未完成条数 |\n")
    sections.append("|-----:|---------:|-------------:|-----------|---------:|-----------:|\n")
    for r in turn_records:
        ids = ", ".join(r["sealed_ids"]) if r["sealed_ids"] else "—"
        sections.append(
            f"| {r['turn']} | {len(r['input'])} | {r['sealed_count']} | {ids} | "
            f"{r['shadow_len']} | {r['unclosed_count']} |\n"
        )

    body = "".join(sections)
    _write_live_report_md(report_path, sections, progress_turn=None, total_turns=n_turns)

    assert report_path.is_file()
    assert len(body) > 200, "Report unexpectedly short — check LLM / pipeline errors above."
    print(f"\n[live-metabolism] 完整报告: {report_path.resolve()}\n", flush=True)
