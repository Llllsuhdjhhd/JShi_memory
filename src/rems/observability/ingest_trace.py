from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STEP_LABELS: dict[str, str] = {
    "shadow_load": "残影加载",
    "recall_query_compress": "检索 query 压缩",
    "pre_recall_role_extract": "Pre-recall 角色提交",
    "role_extract_await": "角色抽取",
    "recall_build_context": "回忆块组装",
    "recall_log_append": "recall_log 登记",
    "recall_relevance_audit": "回忆相关性审计",
    "metabolism_seal": "代谢封存",
    "role_timeline_update": "角色白描更新",
    "abstraction_mine": "抽象归纳",
    "ingest_complete": "ingest 完成",
}

# LLM task_type → 控制台展示名（与 Skill 对齐）
LLM_TASK_LABELS: dict[str, str] = {
    "recall_query_compress": "RecallQueryCompressSkill",
    "role_extraction": "RoleExtractionSkill",
    "boundary_detection": "BoundaryDetectionSkill",
    "event_enrichment": "EventEnrichmentSkill",
    "narrative_coherence": "NarrativeCoherenceSkill",
    "abstraction": "AbstractionSkill",
}


def event_preview(event: Any, *, summary_max: int = 60) -> dict[str, Any]:
    """Lightweight event summary for trace logs."""
    summary = ""
    summaries = getattr(event, "summaries", None) or {}
    if isinstance(summaries, dict):
        summary = str(summaries.get("L1") or summaries.get("l1") or "")[:summary_max]
    role_list = getattr(event, "role_list", None) or []
    return {
        "event_id": getattr(event, "event_id", None),
        "summary_l1": summary,
        "chars": getattr(event, "event_length", None) or len(getattr(event, "content_raw", "") or ""),
        "roles": [getattr(r, "role_id", None) for r in role_list[:6]],
        "is_abstract": bool(getattr(event, "is_abstract", False)),
    }


@dataclass
class IngestTrace:
    """Optional step-by-step trace for ``REMSPipeline.ingest`` (scenario diagnostics)."""

    echo: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    _current_step: str | None = field(default=None, repr=False)
    _pending_llm: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def set_current_step(self, step: str | None) -> None:
        self._current_step = step
        self._pending_llm = []
        if not self.echo or not step:
            return
        label = STEP_LABELS.get(step, step)
        print(f"  [Pipeline] > {label}", flush=True)

    def record_llm(
        self,
        *,
        task_type: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        log_file: str | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "task_type": task_type,
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "pipeline_step": self._current_step,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "log_file": log_file,
        }
        self._pending_llm.append(rec)
        if not self.echo:
            return
        tok = _format_tokens(prompt_tokens, completion_tokens, total_tokens)
        file_hint = f" → {log_file}" if log_file else ""
        skill = LLM_TASK_LABELS.get(task_type, task_type)
        print(
            f"      [Skill] {skill} | {task_type} | {model} | {latency_ms:.0f}ms | {tok}{file_hint}",
            flush=True,
        )

    def record(self, step: str, elapsed_ms: float, **detail: Any) -> None:
        llm_calls = list(self._pending_llm)
        if llm_calls:
            detail = {**detail, "llm_calls": llm_calls}
        rec: dict[str, Any] = {
            "step": step,
            "label": STEP_LABELS.get(step, step),
            "elapsed_ms": round(elapsed_ms, 2),
            **detail,
        }
        self.steps.append(rec)
        self._pending_llm = []
        if self._current_step == step:
            self._current_step = None
        if not self.echo:
            return
        label = rec["label"]
        ms = rec["elapsed_ms"]
        summary = _format_detail({k: v for k, v in detail.items() if k != "llm_calls"})
        llm_hint = f", LLM×{len(llm_calls)}" if llm_calls else ""
        suffix = f" — {summary}" if summary else ""
        print(f"  [Pipeline] OK {label}: {ms:.0f}ms{llm_hint}{suffix}", flush=True)


def _format_tokens(
    prompt: int | None,
    completion: int | None,
    total: int | None,
) -> str:
    if prompt is None and completion is None and total is None:
        return "tokens n/a"
    if total is not None:
        return f"tokens {total} (in={prompt or '?'} out={completion or '?'})"
    parts = []
    if prompt is not None:
        parts.append(f"in={prompt}")
    if completion is not None:
        parts.append(f"out={completion}")
    return "tokens " + " ".join(parts) if parts else "tokens n/a"


def _format_detail(detail: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in detail.items():
        if value is None:
            continue
        if key == "sealed" and isinstance(value, list):
            parts.append(f"封存 {len(value)} 条")
            continue
        if key == "abstracts" and isinstance(value, list):
            parts.append(f"新抽象 {len(value)} 条")
            continue
        if key == "recall_items":
            parts.append(f"回忆块 {value} 条")
            continue
        if key == "extracted_roles":
            parts.append(f"抽取 {value} 角色")
            continue
        if key == "act_chars":
            parts.append(f"act摘要 {value} 字")
            continue
        if key == "act_query_preview":
            if value:
                parts.append(f"act={value}")
            continue
        if key == "compress_skipped":
            if value:
                parts.append("跳过LLM压缩")
            continue
        if key == "compress_source":
            parts.append(f"来源={value}")
            continue
        if key == "focus_role_ids":
            parts.append(f"focus {len(value)} 个" if isinstance(value, (list, set)) else f"focus {value}")
            continue
        if key == "shadow_chars":
            parts.append(f"残影 {value} 字")
            continue
        if key == "unclosed_count":
            parts.append(f"未完成 {value} 条")
            continue
        if key == "unclosed_after":
            parts.append(f"剩余未完成 {value} 条")
            continue
        if key == "n_events":
            parts.append(f"登记 {value} 事件")
            continue
        if key == "events_updated":
            parts.append(f"更新 {value} 事件")
            continue
        if key == "input_chars":
            parts.append(f"输入 {value} 字")
            continue
        if key == "llm_calls":
            continue
        if key == "ran" and value is False:
            continue
        if key in ("event_ids", "role_ids", "role_names"):
            if isinstance(value, list) and value:
                preview = ", ".join(str(x) for x in value[:4])
                if len(value) > 4:
                    preview += f" …(+{len(value) - 4})"
                parts.append(f"{key}=[{preview}]")
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    return ", ".join(parts)
