import os
import json
import shutil
import time
import sqlite3
import sys
import statistics
import re
from pathlib import Path
from typing import Any

# Ensure all prints are flushed immediately
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline, ProcessingMode
from rems.llm.provider import LLMProvider


def _parse_recall_audit_from_llm_json(path: Path) -> dict[str, Any]:
    """Decode saved LLM log; return partition validity and rate_b used by pipeline semantics."""
    data = json.loads(path.read_text(encoding="utf-8"))
    user_txt = ""
    for msg in data.get("messages") or []:
        if msg.get("role") == "user":
            user_txt = msg.get("content") or ""
            break
    expected_ids = frozenset(re.findall(r"event_id `([^`]+)`", user_txt))
    rsp = (data.get("response") or "").strip()
    if "```" in rsp:
        inner = rsp
        if inner.startswith("```"):
            inner = inner.split("\n", 1)[-1]
        inner = inner.rsplit("```", 1)[0].strip()
        if inner.startswith("json"):
            inner = inner.split("\n", 1)[-1].strip()
        rsp = inner
    try:
        obj = json.loads(rsp)
    except json.JSONDecodeError as e:
        return {"path": str(path.name), "parse_error": str(e), "expected_ids_n": len(expected_ids)}
    rel = frozenset(str(x) for x in (obj.get("related_event_ids") or []) if x)
    unr = frozenset(str(x) for x in (obj.get("unrelated_event_ids") or []) if x)
    valid = not (rel - expected_ids or unr - expected_ids or rel & unr or (rel | unr) != expected_ids)
    rate = len(rel) / max(1, len(expected_ids)) if valid else None
    return {
        "path": str(path.name),
        "expected_ids_n": len(expected_ids),
        "partition_valid": valid,
        "related_n": len(rel),
        "unrelated_n": len(unr),
        "rate_b_effective_pipeline": 0.5 if not valid else rate,
        "rate_b_if_partition_valid": rate,
    }


def _enrich_chunks_with_b_audit_logs(log_dir: Path, chunks_diag: list[dict]) -> None:
    for entry in chunks_diag:
        cid = entry.get("chunk_idx")
        matches = sorted(log_dir.glob(f"chunk_{cid}_*_recall_block_relevance.json"))
        if matches:
            entry["rate_b_audit_log"] = _parse_recall_audit_from_llm_json(matches[-1])


def _write_recall_cap_markdown_report(base_dir: Path, diag_payload: dict) -> Path:
    lines: list[str] = []
    lines.append("# 单块向量 distance · semantic_cap · 回忆相关性 **b** 诊断报告")
    lines.append("")
    lines.append(
        "由 `continuous_simulation.py` 写入 `recall_cap_diagnostic.json` 后生成；"
        "并解析 `llm_calls/chunk_*_*_recall_block_relevance.json`（相对目录 `continuous_run/`）。"
    )
    lines.append("")
    cs = diag_payload.get("config_snapshot") or {}
    chunks = diag_payload.get("chunks") or []
    lines.append("## 1. 流水线顺序与 **b** 对 cap 的生效时刻")
    lines.append("")
    lines.append(
        "- **先回忆**：用当前的 `combined` 得到 `semantic_cap`，按 **distance ≤ cap** 过滤向量命中。"
    )
    lines.append(
        "- **再触发 b**：`record_recall_block_relevance_rate` 更新 **`ema_b`**（由 **rate_b** 驱动）。"
    )
    lines.append(
        "- 因此**同一块**内：**b** 不改变当次检索；只影响「**b** 之后立刻再回忆」或**下一块**的 cap。"
    )
    lines.append("")
    lines.append("## 2. 配置快照")
    lines.append("")
    for k, v in sorted(cs.items()):
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## 3. 双流 distance 与 cap")
    lines.append("")
    lines.append(diag_payload.get("distance_note", ""))
    lines.append("")
    for ent in chunks:
        cid = ent.get("chunk_idx")
        lines.append(f"### Chunk `{cid}`")
        pre = ent.get("pre_recall_quality") or {}
        lines.append("")
        lines.append("| 回忆拼装前 | 值 |")
        lines.append("|---|-----|")
        for k in ("combined", "semantic_cap", "ema_a", "ema_b"):
            lines.append(f"| {k} | {pre.get(k)} |")
        lines.append("")
        lines.append("| Stream | raw_n | kept_n | dropped | dist>cap | cap | d_min | d_median | d_max |")
        lines.append("|--------|-------|--------|---------|----------|-----|-------|----------|-------|")
        for srow in ent.get("streams") or []:
            lines.append(
                f"| {srow.get('stream')} | {srow.get('n_hits_raw')} | {srow.get('n_hits_kept')} | "
                f"{srow.get('n_dropped_gt_cap')} | {srow.get('n_distance_gt_cap')} | {srow.get('cap_applied')} | "
                f"{srow.get('distance_min')} | {srow.get('distance_median')} | {srow.get('distance_max')} |"
            )
        lines.append("")
        lines.append(f"- `recall_block` 条目数: **{ent.get('recall_block_item_count')}**")
        post = ent.get("post_ingest_after_b_audit") or {}
        if post:
            lines.append("")
            lines.append("| **b** 写入后（立即再算 cap） | 值 |")
            lines.append("|---|-----|")
            lines.append(f"| ema_b | {post.get('ema_b')} |")
            lines.append(f"| combined | {post.get('combined')} |")
            lines.append(f"| semantic_cap_now | {post.get('semantic_cap_if_recall_ran_now')} |")
            c0, c1 = pre.get("semantic_cap"), post.get("semantic_cap_if_recall_ran_now")
            try:
                if c0 is not None and c1 is not None:
                    lines.append("")
                    lines.append(f"- Δcap（回忆用 → **b** 后）: **{float(c1) - float(c0):.6f}**")
            except (TypeError, ValueError):
                pass
        audit = ent.get("rate_b_audit_log")
        if audit:
            lines.append("")
            lines.append("#### `recall_block_relevance` 日志解析")
            lines.append("")
            lines.append("| 字段 | 值 |")
            lines.append("|------|-----|")
            for ak, av in audit.items():
                lines.append(f"| {ak} | {av} |")
        lines.append("")
    sens = float(cs.get("recall_dynamic_distance_sensitivity") or 0)
    base_v = float(cs.get("recall_dynamic_distance_cap_base") or 0)
    ceil_v = float(cs.get("recall_dynamic_distance_cap_ceiling") or 0)
    lines.append("## 4. 调节幅度是否偏保守")
    lines.append("")
    lines.append(
        f"- **`sensitivity = {sens}`**：`combined` 相对 neutral 偏移 0.5 量级时，`adjusted` 偏移约 **`±{sens * 0.5:.4f}`**（未触顶触底前）。"
        "相对 distance ~0–2 的标尺，单凭此项往往只是微调。"
    )
    if abs(ceil_v - base_v) < 1e-6:
        lines.append(
            f"- **`cap_ceiling == cap_base ({base_v})`**：`combined ≥ neutral` 时 cap **无法再高**，只能靠低 combined **略收紧**。"
        )
    lines.append(
        "- 若表中 **dropped** 长期为 0，说明 **cap** 远高于本批 **`d_max`**，需 **显著降低 cap_base**或**抬高 sensitivity**，或让 **ceiling > base**，否则 **b→cap** 链路上「看得见但裁不动」。"
    )
    lines.append("")
    lines.append("### 可调键（不改 ingest 主干）")
    lines.append("")
    lines.append("| 目标 | `REMSConfig` 键 |")
    lines.append("|------|-----------------|")
    lines.append("| combined 动动 cap 更明显 | ↑ `recall_dynamic_distance_sensitivity` |")
    lines.append("| 质量好时略微放宽尾部 | `recall_dynamic_distance_cap_ceiling` > base |")
    lines.append("| 质量差时狠裁 | ↓ base 或 ↑ sens；必要时 ↓ floor |")
    lines.append("")
    lines.append("## 5. cap 与各流 distance 上限（自动生成）")
    lines.append("")
    for ent in chunks:
        cid = ent.get("chunk_idx")
        post = ent.get("post_ingest_after_b_audit") or {}
        cap_now = post.get("semantic_cap_if_recall_ran_now")
        if cap_now is None:
            lines.append(f"- Chunk `{cid}`：无 post_ingest **`semantic_cap_if_recall_ran_now`**（未写入 **b**？）")
            lines.append("")
            continue
        try:
            cn = float(cap_now)
        except (TypeError, ValueError):
            cn = None
        if cn is None:
            lines.append("")
            continue
        lines.append(f"### Chunk `{cid}`：**b** 写入后的 hypothetic cap = **{cn:.6f}**")
        for srow in ent.get("streams") or []:
            tag = srow.get("stream")
            dmx = srow.get("distance_max")
            try:
                dmf = float(dmx) if dmx is not None else None
            except (TypeError, ValueError):
                dmf = None
            if dmf is None:
                lines.append(f"- Stream **{tag}**：无 **`d_max`**。")
                continue
            slack = cn - dmf
            verdict = "**仍高于**本流 `d_max`，本批 hit 不会因该 cap 被裁。" if slack > 1e-9 else "已与尾部 hit 接壤或更严，可望出现剔除。"
            lines.append(f"- Stream **{tag}**：`d_max`≈**{dmf:.6f}**，`cap_now−d_max`≈**{slack:.6f}** ⇒ {verdict}")
        audit = ent.get("rate_b_audit_log") or {}
        if audit.get("partition_valid") is True:
            lines.append(
                f"- **`rate_b`**：{audit.get('rate_b_effective_pipeline')}（related **{audit.get('related_n')}** / "
                f"expected **{audit.get('expected_ids_n')}**）。"
            )
        lines.append("")
    lines.append(
        "**解读**：若在 **b** 之后 cap 仍远高于两路 **`d_max`**，则 **`sensitivity`、`base`、`ceiling` 再怎么微调也可能「零剔除」**；需把 **`cap_base`（或等价标尺）拉近 hit 的典型 distance**。"
    )
    lines.append("")
    lines.append("`llm_calls/` 路径: `tests/scenarios/hongloumeng/outputs/continuous_run/llm_calls/`")
    lines.append("")
    out_path = base_dir / "RECALL_CAP_B_DISTANCE_REPORT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_continuous_simulation(num_chunks_to_process=3):
    """
    Continuous simulation runner that preserves state between runs.
    """
    base_dir = Path(__file__).parent / "outputs" / "continuous_run"
    base_dir.mkdir(parents=True, exist_ok=True)

    state_file = base_dir / "simulation_state.json"
    db_path = base_dir / "rems_sim.db"
    chroma_path = base_dir / "chroma_sim"
    log_dir = base_dir / "llm_calls"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load state
    last_chunk_idx = 0
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                last_chunk_idx = state.get("last_chunk_idx", 0)
        except Exception as e:
            print(f"Warning: Could not load state file: {e}")

    # Initialize REMS
    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{db_path}",
            qdrant_path=str(chroma_path)
        ),
        embedding={"provider": "hash"},
        user_mode=UserMode.MULTI
    )
    # 单块冒烟：`ingest_seq` 仅从 1 递增一次，默认 every_n=10 不会触发 rate **b**；此处强制首轮即抽检。
    # 同时打开动态语义距离，否则 cap=None，无法记录「向量 distance 与 cap」关系（**b** 仍会在本块末尾更新 EMA，供下一块使用）。
    if num_chunks_to_process == 1:
        config.recall_relevance_audit_every_n_ingests = 1
        config.recall_dynamic_distance_enabled = True
        config.recall_dynamic_distance_hit_percentile = 80.0
        print(
            "  [CFG] single-chunk: recall_relevance_audit_every_n_ingests=1, "
            "recall_dynamic_distance_enabled=True, recall_dynamic_distance_hit_percentile=80 (b + cap/distance diagnostics)"
        )
    # 双块 regression：两块内每轮 ingest 都做「回忆块相关性」评估 (**b**) ，并打开动态语义距离裁剪，
    # 这样第二块的回忆双流才会体现第一块 ingest 末尾刷新后的 combined(**a**/ **b**) → cap。
    if num_chunks_to_process == 2:
        config.recall_relevance_audit_every_n_ingests = 1
        config.recall_dynamic_distance_enabled = True
        config.recall_dynamic_distance_hit_percentile = 80.0
        print(
            "  [CFG] 2-chunk AB/recall probe: recall_relevance_audit_every_n_ingests=1, "
            "recall_dynamic_distance_enabled=True, recall_dynamic_distance_hit_percentile=80"
        )

    # Setup LLM logging (similar to simulation.py)
    orig_complete = LLMProvider.complete
    call_counter = 0

    def complete_with_logging(self, task_type, messages, **kwargs):
        nonlocal call_counter
        call_counter += 1
        model = self._get_model(task_type)
        print(f"  [LLM] #{call_counter:03} {task_type:.<18} | {model:.<20}", end="", flush=True)

        t_start = time.perf_counter()
        content = orig_complete(self, task_type, messages, **kwargs)
        duration_ms = (time.perf_counter() - t_start) * 1000
        metrics = self._invocations[-1]

        # Log detail
        log_file = log_dir / f"chunk_{last_chunk_idx + 1}_{call_counter:03}_{task_type}.json"
        from dataclasses import asdict
        log_data = {
            "chunk_idx": last_chunk_idx + 1,
            "task_type": task_type,
            "model": model,
            "messages": messages,
            "response": content,
            "metrics": asdict(metrics)
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f" DONE ({duration_ms:4.0f}ms)")
        return content

    LLMProvider.complete = complete_with_logging

    pipeline = REMSPipeline.from_config(config)

    recall_cap_chunks: list[dict] = []
    _diag_chunk_idx = [0]
    _capture_recall_cap_diag = getattr(config, "recall_dynamic_distance_enabled", False) or (
        config.recall_relevance_audit_every_n_ingests <= 1
    )
    if _capture_recall_cap_diag:
        # 双流向量 distance 与 cap、「**b** 之后」cap（写入 recall_cap_diagnostic.json + Markdown 报告）。
        _filter_call_in_block = [0]
        _rs = pipeline.recall_service
        _orig_filter_hits = _rs._filter_hits_semantic_distance
        _diag_chunk_idx = [0]

        def _filter_hits_with_distance_stats(hits):
            cap = getattr(_rs, "_stream_distance_cap_effective", None)
            if cap is None:
                rq2 = getattr(pipeline, "recall_quality", None)
                cap = rq2.semantic_distance_cap() if rq2 is not None else None
            _filter_call_in_block[0] += 1
            tag = "A" if _filter_call_in_block[0] == 1 else "B"
            distances = [float(h.get("distance", 1.0)) for h in hits] if hits else []
            n_in = len(distances)
            n_gt_cap = (
                sum(1 for d in distances if cap is not None and d > cap + 1e-9)
                if cap is not None
                else 0
            )
            kept = _orig_filter_hits(hits)
            n_out = len(kept)
            row = {
                "chunk_idx": _diag_chunk_idx[0],
                "stream": tag,
                "cap_applied": cap,
                "n_hits_raw": n_in,
                "n_hits_kept": n_out,
                "n_dropped_gt_cap": n_in - n_out,
                "n_distance_gt_cap": n_gt_cap,
                "distance_min": min(distances) if distances else None,
                "distance_max": max(distances) if distances else None,
                "distance_mean": (sum(distances) / len(distances)) if distances else None,
                "distance_median": (statistics.median(distances) if distances else None),
            }
            if recall_cap_chunks and recall_cap_chunks[-1].get("chunk_idx") == _diag_chunk_idx[0]:
                recall_cap_chunks[-1]["streams"].append(row)
            return kept

        _rs._filter_hits_semantic_distance = _filter_hits_with_distance_stats

        _orig_build_recall = _rs.build_recall_block

        def _build_recall_block_logged(*args, **kwargs):
            _filter_call_in_block[0] = 0
            rq = getattr(pipeline, "recall_quality", None)
            ema_a = ema_b = combined = cap = None
            if rq is not None:
                ema_a = getattr(rq, "_ema_a", None)
                ema_b = getattr(rq, "_ema_b", None)
                combined = rq.combined_signal()
                cap = rq.semantic_distance_cap()
            snapshot = {
                "combined": combined,
                "semantic_cap": cap,
                "ema_a": ema_a,
                "ema_b": ema_b,
            }
            if not recall_cap_chunks or recall_cap_chunks[-1]["chunk_idx"] != _diag_chunk_idx[0]:
                recall_cap_chunks.append({"chunk_idx": _diag_chunk_idx[0], "pre_recall_quality": snapshot, "streams": []})
            else:
                recall_cap_chunks[-1]["pre_recall_quality"] = snapshot
            block = _orig_build_recall(*args, **kwargs)
            audit = getattr(_rs, "recall_distance_cap_audit_snapshot", None)
            if audit:
                pq = recall_cap_chunks[-1]["pre_recall_quality"]
                pq["effective_cap_used_for_filter"] = audit.get("effective_cap")
                pq["ema_semantic_cap"] = audit.get("ema_semantic_cap")
                pq["percentile_cap_after_clamp"] = audit.get("percentile_cap_after_clamp")
                pq["hit_percentile_config"] = audit.get("hit_percentile_config")
                pq["hit_distances_count"] = audit.get("hit_distances_count")
                recall_cap_chunks[-1]["pre_recall_quality"] = pq
            n = len(block.items) if getattr(block, "items", None) is not None else 0
            recall_cap_chunks[-1]["recall_block_item_count"] = n
            eff_u = (
                recall_cap_chunks[-1].get("pre_recall_quality") or snapshot
            ).get("effective_cap_used_for_filter")
            print(
                f"  [RECALL] items={n} combined={combined} ema_cap={cap} effective_cap={eff_u} "
                f"ema_a={ema_a} ema_b={ema_b}"
            )
            return block

        _rs.build_recall_block = _build_recall_block_logged

    # Load dataset
    dataset_path = Path(__file__).parents[3] / "data" / "hongloumeng_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    start_idx = last_chunk_idx
    end_idx = min(start_idx + num_chunks_to_process, len(chunks))

    print(f"\n{'='*60}")
    print(f" REMS CONTINUOUS SIMULATION: CHUNKS {start_idx + 1} TO {end_idx} ")
    print(f"{'='*60}")
    print(f"DB Path: {db_path}")

    # Get current shadow for display
    current_shadow = pipeline.meta_repo.get_shadow()
    print(f"Initial Shadow Length: {len(current_shadow.content)} chars")

    for i in range(start_idx, end_idx):
        chunk = chunks[i]
        content = chunk['content']
        _diag_chunk_idx[0] = i + 1
        print(f"\n--- PROCESSING CHUNK {i+1} (Length: {len(content)}) ---")
        print(f"  [DEBUG] Pipeline file: {pipeline.__class__.ingest.__code__.co_filename}")

        t_chunk_start = time.perf_counter()
        result = pipeline.ingest(content, mode=ProcessingMode.DIALOGUE)
        duration = time.perf_counter() - t_chunk_start

        print(f"  [OK] Duration: {duration:.2f}s")
        print(f"  [Events] Sealed: {len(result.sealed_events)}, Abstracted: {len(result.abstract_events)}")

        rq = getattr(pipeline, "recall_quality", None)
        if rq is not None and recall_cap_chunks and recall_cap_chunks[-1].get("chunk_idx") == i + 1:
            recall_cap_chunks[-1]["post_ingest_after_b_audit"] = {
                "ema_a": getattr(rq, "_ema_a", None),
                "ema_b": getattr(rq, "_ema_b", None),
                "combined": rq.combined_signal(),
                "semantic_cap_if_recall_ran_now": rq.semantic_distance_cap(),
            }
            cap0 = recall_cap_chunks[-1].get("pre_recall_quality", {}).get("semantic_cap")
            cap1 = recall_cap_chunks[-1]["post_ingest_after_b_audit"]["semantic_cap_if_recall_ran_now"]
            pct_val = getattr(config, "recall_dynamic_distance_hit_percentile", None)
            pct_txt = str(pct_val) if pct_val is not None else "off"
            print(
                f"  [RECALL-Q] after-b: combined={recall_cap_chunks[-1]['post_ingest_after_b_audit']['combined']} "
                f"semantic_distance_cap_ema_only={cap1} (recall_used_ema_cap={cap0}); "
                f"下一轮回忆 effective=min(P{pct_txt}_pool,此项EMA)"
            )

        # Update and save state
        last_chunk_idx = i + 1
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"last_chunk_idx": last_chunk_idx}, f, indent=2)

    total_events = pipeline.event_repo.count()
    final_shadow = pipeline.meta_repo.get_shadow()

    print(f"\n{'='*60}")
    print(f" RUN SUMMARY ")
    print(f"{'='*60}")
    print(f"Next Chunk to Process: {last_chunk_idx + 1}")
    print(f"Total Events in DB:    {total_events}")
    print(f"Final Shadow Length:   {len(final_shadow.content)} chars")
    print(f"Final Shadow Preview:  {final_shadow.content[:100]}...")

    if recall_cap_chunks:
        _enrich_chunks_with_b_audit_logs(log_dir, recall_cap_chunks)
        diag_path = base_dir / "recall_cap_diagnostic.json"
        payload = {
            "distance_note": (
                "Stream hit distance 与同库 VectorStore 一致（hash 嵌入下常为 1-cos）；"
                "越小越相似。保留命中需 distance <= semantic_cap（cap 为 None 时不裁剪）。"
            ),
            "config_snapshot": {
                "recall_dynamic_distance_enabled": config.recall_dynamic_distance_enabled,
                "recall_dynamic_distance_cap_base": config.recall_dynamic_distance_cap_base,
                "recall_dynamic_distance_cap_floor": config.recall_dynamic_distance_cap_floor,
                "recall_dynamic_distance_cap_ceiling": config.recall_dynamic_distance_cap_ceiling,
                "recall_dynamic_distance_sensitivity": config.recall_dynamic_distance_sensitivity,
                "recall_dynamic_distance_quality_neutral": config.recall_dynamic_distance_quality_neutral,
                "recall_quality_ema_alpha": config.recall_quality_ema_alpha,
                "recall_quality_weight_coherence_vs_relevance": config.recall_quality_weight_coherence_vs_relevance,
                "recall_dynamic_distance_hit_percentile": config.recall_dynamic_distance_hit_percentile,
            },
            "chunks": recall_cap_chunks,
        }
        with open(diag_path, "w", encoding="utf-8") as df:
            json.dump(payload, df, ensure_ascii=False, indent=2)
        print(f"\n  [DIAG] Wrote {diag_path}")

        md_path = _write_recall_cap_markdown_report(base_dir, payload)
        print(f"  [DIAG] Wrote {md_path}")

if __name__ == "__main__":
    n = 3
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except:
            pass
    run_continuous_simulation(n)
