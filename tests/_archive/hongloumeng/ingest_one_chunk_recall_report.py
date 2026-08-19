"""跑「下一块」红楼梦 ingest，并输出回忆追踪 Markdown + JSON。

与 ``continuous_simulation.py`` 共用 ``outputs/continuous_run``、同一套 DB/state；
额外挂载 ``recall_trace_util.install_recall_hooks``，写：

- ``outputs/continuous_run/recall_last_ingest_report.md``
- ``outputs/continuous_run/recall_last_ingest_trace.json``

用法（仓库根目录）::

    conda run -n py3125 python tests/scenarios/hongloumeng/ingest_one_chunk_recall_report.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.llm.provider import LLMProvider
from rems.pipeline import REMSPipeline, ProcessingMode

from recall_trace_util import (
    install_recall_hooks,
    latest_recall_log_row,
    recall_block_to_json,
    write_recall_trace_json,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for d in [here, *here.parents]:
        if (d / "data" / "hongloumeng_dataset.json").is_file():
            return d
    raise FileNotFoundError(
        "无法在脚本上级目录中找到 data/hongloumeng_dataset.json，请从 REMS 仓库根运行。"
    )


def _md_fence(title: str, body: str, limit: int = 12000) -> str:
    text = body if len(body) <= limit else body[:limit] + "\n\n…（截断）…"
    return f"### {title}\n\n```text\n{text}\n```\n\n"


def _hit_rows_md(hits: list[dict], n: int = 12) -> str:
    lines = ["| # | event_id / wp_id | distance | preview |", "|---|------------------|----------|---------|"]
    for i, h in enumerate(hits[:n], start=1):
        eid = h.get("event_id") or ""
        wid = h.get("wp_id") or ""
        dist = h.get("distance", "")
        prev = (h.get("document_preview") or h.get("document") or "")[:180].replace("|", "\\|")
        key = eid or wid
        lines.append(f"| {i} | `{key}` | {dist} | {prev} |")
    return "\n".join(lines) + "\n\n"


def main() -> int:
    scenario_dir = Path(__file__).resolve().parent
    base_dir = scenario_dir / "outputs" / "continuous_run"
    base_dir.mkdir(parents=True, exist_ok=True)

    state_file = base_dir / "simulation_state.json"
    db_path = base_dir / "rems_sim.db"
    chroma_path = base_dir / "chroma_sim"
    log_dir = base_dir / "llm_calls"
    log_dir.mkdir(parents=True, exist_ok=True)

    last_chunk_idx = 0
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                last_chunk_idx = int(json.load(f).get("last_chunk_idx", 0))
        except Exception as exc:
            print(f"Warning: state load failed: {exc}", flush=True)

    dataset_path = _repo_root() / "data" / "hongloumeng_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    i = last_chunk_idx
    if i >= len(chunks):
        print(f"No chunk left (last_chunk_idx={i}, len={len(chunks)})", flush=True)
        return 1

    chunk = chunks[i]
    content = chunk["content"]

    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{db_path}",
            qdrant_path=str(chroma_path),
        ),
        embedding={"provider": "hash"},
        user_mode=UserMode.MULTI,
    )

    orig_complete = LLMProvider.complete
    call_counter = 0

    def complete_with_logging(self, task_type, messages, **kwargs):
        nonlocal call_counter
        call_counter += 1
        model = self._get_model(task_type)
        print(f"  [LLM] #{call_counter:03} {task_type:.<18} | {model:.<20}", end="", flush=True)
        t0 = time.perf_counter()
        out = orig_complete(self, task_type, messages, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        metrics = self._invocations[-1]
        log_file = log_dir / f"chunk_{i + 1}_{call_counter:03}_{task_type}.json"
        payload = {
            "chunk_idx": i + 1,
            "task_type": task_type,
            "model": model,
            "messages": messages,
            "response": out,
            "metrics": asdict(metrics),
        }
        with open(log_file, "w", encoding="utf-8") as lf:
            json.dump(payload, lf, ensure_ascii=False, indent=2)
        print(f" DONE ({ms:5.0f}ms)", flush=True)
        return out

    pipeline = REMSPipeline.from_config(config)
    shadow_before = pipeline.meta_repo.get_shadow().content or ""

    trace, uninstall_recall = install_recall_hooks(pipeline.recall_service)

    md_parts: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    md_parts.append(f"# 红楼梦 ingest — 回忆追踪报告\n\n")
    md_parts.append(f"- 生成时间: {ts}\n")
    md_parts.append(f"- 数据集块: **{i + 1}** / {len(chunks)}（内部索引 `i={i}`）\n")
    md_parts.append(f"- DB: `{db_path}`\n\n")
    md_parts.append(_md_fence("本轮输入（节选）", content[:2500]))
    md_parts.append(_md_fence("ingest 前残影 Shadow（节选）", shadow_before[:2500]))

    print(f"\n{'='*60}\n CHUNK {i + 1} (chars={len(content)}) + recall trace\n{'='*60}\n", flush=True)

    try:
        LLMProvider.complete = complete_with_logging
        t_chunk = time.perf_counter()
        try:
            result = pipeline.ingest(content, mode=ProcessingMode.DIALOGUE)
        finally:
            uninstall_recall()
        dur = time.perf_counter() - t_chunk
    finally:
        LLMProvider.complete = orig_complete

    md_parts.append(f"## Ingest 结果\n\n")
    md_parts.append(f"- 耗时: **{dur:.2f}s**\n")
    md_parts.append(f"- Sealed: **{len(result.sealed_events)}** · Abstracted: **{len(result.abstract_events)}**\n\n")

    n_rounds = len(trace["stream_a"])
    md_parts.append(f"## 双流召回快照（本轮共 **{n_rounds}** 次 `build_recall_block` 内部回合）\n\n")

    for r in range(n_rounds):
        md_parts.append(f"### 回合 {r + 1}\n\n")
        if r < len(trace["stream_a"]):
            sa = trace["stream_a"][r]
            md_parts.append(_md_fence("Stream A · query 前缀", sa.get("query_prefix", "")))
            md_parts.append(f"**Stream A · hits（前 12）**\n\n{_hit_rows_md(sa.get('hits') or [])}")
        if r < len(trace["stream_b"]):
            sb = trace["stream_b"][r]
            md_parts.append(_md_fence("Stream B · query 前缀", sb.get("query_prefix", "")))
            md_parts.append(f"**Stream B · hits（前 12）**\n\n{_hit_rows_md(sb.get('hits') or [])}")
        if r < len(trace["rrf"]):
            mt = trace["rrf"][r].get("merged_top") or []
            md_parts.append("**RRF merged_top**\n\n")
            md_parts.append("| # | event_id | score |\n|---|----------|-------|\n")
            for j, row in enumerate(mt[:24], start=1):
                md_parts.append(f"| {j} | `{row.get('event_id', '')}` | {row.get('score', '')} |\n")
            md_parts.append("\n")

    rb_json: list = []
    if result.context_package and result.context_package.recall_block:
        rb = result.context_package.recall_block
        rb_json = recall_block_to_json(rb)
        md_parts.append("## 最终进入上下文的 Recall Block\n\n")
        md_parts.append("| # | event_id | level | score | content_preview |\n")
        md_parts.append("|---|----------|-------|-------|-----------------|\n")
        for j, it in enumerate(rb_json, start=1):
            prev = (it.get("content_preview") or "").replace("|", "\\|")[:200]
            md_parts.append(
                f"| {j} | `{it.get('event_id', '')}` | {it.get('summary_level', '')} | "
                f"{it.get('score', '')} | {prev} |\n"
            )
        md_parts.append("\n")
    else:
        md_parts.append("## 最终 Recall Block\n\n_（本轮 `context_package` 为空或无 recall_block）_\n\n")

    log_last = latest_recall_log_row(pipeline.recall_log_repo)
    md_parts.append("## recall_log 最新一行\n\n")
    md_parts.append(f"```json\n{json.dumps(log_last, ensure_ascii=False, indent=2)}\n```\n\n")

    shadow_after = pipeline.meta_repo.get_shadow().content or ""
    md_parts.append(_md_fence("ingest 后残影 Shadow（节选）", shadow_after[:2500]))

    md_parts.append(
        f"## 机器可读\n\n"
        f"- 完整 trace: `{base_dir / 'recall_last_ingest_trace.json'}`\n"
        f"- LLM 明细目录: `{log_dir}`\n"
    )

    report_md = base_dir / "recall_last_ingest_report.md"
    report_md.write_text("".join(md_parts), encoding="utf-8")

    write_recall_trace_json(
        base_dir / "recall_last_ingest_trace.json",
        {
            "kind": "ingest_one_chunk_recall_report",
            "chunk_1based": i + 1,
            "stream_a": trace["stream_a"],
            "stream_b": trace["stream_b"],
            "rrf": trace["rrf"],
            "recall_block": rb_json,
            "recall_log_last": log_last,
            "sealed_event_ids": [e.event_id for e in result.sealed_events],
            "abstract_event_ids": [e.event_id for e in result.abstract_events],
        },
    )

    last_chunk_idx = i + 1
    with open(state_file, "w", encoding="utf-8") as sf:
        json.dump({"last_chunk_idx": last_chunk_idx}, sf, indent=2)

    total_events = pipeline.event_repo.count()
    print(f"\n{'='*60}\n DONE · next chunk={last_chunk_idx + 1} · events_in_db={total_events}\n", flush=True)
    print(f"Report: {report_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
