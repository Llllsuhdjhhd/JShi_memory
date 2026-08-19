"""Minimal metrics scaffolding for ``tests/integration/test_wp_live_*`` suites.

历史背景：旧仓库下这里有完整 ``InstrumentationStore``/``WPReporter``/``probes``/``live_common`` 实现，
用于 LLM-live 集成测试期间的"观测 + 报告生成"。架构升级后整套被外迁，wp_live 系列测试只在
``REMS_RUN_LIVE_METABOLISM_TEST=1`` 时才跑（默认 ``pytest`` 跳过），但 import 阶段仍会触达
``from .metrics import (...)`` —— 没有这个包就直接 ``ImportError`` 阻塞整套 collection。

本骨架只保证以下两件事：

1. **collection 不炸**：所有上层 import 的符号都有占位实现（dataclass 或 no-op callable）；
2. **live_llm_skip_if_disabled() 提前跳过**：默认环境下 wp_live 测试在第一行就被 ``pytest.skip``，
   不会真正执行 ``probe`` / ``WPReporter.finalize`` 等深度逻辑，因而占位实现的"语义薄"是可接受的。

如果未来需要恢复真实的 live 报告，请把这里替换成完整实现并扩充 ``probes`` / ``live_common``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ProbeResult:
    """Lightweight probe outcome (passed / failed / details)."""

    name: str
    passed: bool = True
    details: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstrumentationStore:
    """Capture LLM/vector/scoring traces for assertion in live tests.

    占位实现：只暴露下游测试访问的字段名，所有列表/字典默认空，对应深度逻辑被
    ``live_llm_skip_if_disabled()`` 跳过时不被触达。
    """

    llm_invocations: list[dict[str, Any]] = field(default_factory=list)
    vector_calls: list[dict[str, Any]] = field(default_factory=list)
    hybrid_scores: list[dict[str, Any]] = field(default_factory=list)
    abstraction_user_messages: list[str] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)

    def probe(self, result: ProbeResult) -> ProbeResult:
        self.probes.append(result)
        return result


class WPReporter:
    """No-op markdown reporter shim.

    保持上层调用形态：``set_config_snapshot``、``set_llm_env_summary``、``set_repro_footer``、
    ``section`` (context manager)、``add_run_summary``、``set_llm_invocation_metrics``、
    ``finalize``、``report_path``。所有方法返回自身或 None，``section`` 返回一个支持
    ``with`` 语法的占位上下文管理器。
    """

    def __init__(self, report_path: Path | str, title: str = "") -> None:
        self.report_path = Path(report_path)
        self.title = title

    def set_config_snapshot(self, *_args: Any, **_kw: Any) -> "WPReporter":
        return self

    def set_llm_env_summary(self, *_args: Any, **_kw: Any) -> "WPReporter":
        return self

    def set_repro_footer(self, *_args: Any, **_kw: Any) -> "WPReporter":
        return self

    def add_run_summary(self, *_args: Any, **_kw: Any) -> "WPReporter":
        return self

    def set_llm_invocation_metrics(self, *_args: Any, **_kw: Any) -> "WPReporter":
        return self

    def finalize(self, *_args: Any, **_kw: Any) -> Path:
        return self.report_path

    def section(self, *_args: Any, **_kw: Any) -> "_SectionCtx":
        return _SectionCtx()


class _SectionCtx:
    """Context manager used as ``with rep.section('xxx') as s: s.probe(...)``."""

    def __enter__(self) -> "_SectionCtx":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def probe(self, result: ProbeResult) -> ProbeResult:
        return result

    def method(self, *_args: Any, **_kw: Any) -> None:
        return None

    def add(self, *_args: Any, **_kw: Any) -> None:
        return None


def default_report_path(filename: str) -> Path:
    """Return ``$REMS_WP_REPORT_DIR/<filename>`` or ``./reports/<filename>`` by default."""
    base = Path(os.environ.get("REMS_WP_REPORT_DIR", "reports"))
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


def install_llm_tracer(store: InstrumentationStore, llm: Any) -> Callable[[], None]:
    """Wrap ``llm.complete_json`` to record invocations into ``store``. Returns un-installer."""
    if not hasattr(llm, "complete_json"):
        return lambda: None
    orig = llm.complete_json

    def wrapped(task_type: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        store.llm_invocations.append({"task_type": task_type, "messages": messages})
        if task_type == "inductive_evolution":
            try:
                store.abstraction_user_messages.append(messages[-1]["content"])
            except (IndexError, KeyError, TypeError):
                pass
        return orig(task_type, messages, **kwargs)

    llm.complete_json = wrapped  # type: ignore[assignment]

    def uninstall() -> None:
        llm.complete_json = orig  # type: ignore[assignment]

    return uninstall


def install_vector_tracer(store: InstrumentationStore, vector_store: Any) -> Callable[[], None]:
    """Wrap ``VectorStore.search`` for trace capture."""
    if not hasattr(vector_store, "search"):
        return lambda: None
    orig = vector_store.search

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = orig(*args, **kwargs)
        store.vector_calls.append({"args": args, "kwargs": kwargs, "n_results": len(result)})
        return result

    vector_store.search = wrapped  # type: ignore[assignment]

    def uninstall() -> None:
        vector_store.search = orig  # type: ignore[assignment]

    return uninstall


def install_recall_scoring_tracer(store: InstrumentationStore, recall_service: Any) -> Callable[[], None]:
    """Wrap legacy ``_hybrid_score`` for diagnostic capture (P1-7 改造后只剩诊断兼容意义)."""
    if not hasattr(recall_service, "_hybrid_score"):
        return lambda: None
    orig = recall_service._hybrid_score

    def wrapped(event: Any, cosine_sim: float) -> float:
        score = orig(event, cosine_sim)
        store.hybrid_scores.append(
            {"event_id": getattr(event, "event_id", None), "cosine_sim": cosine_sim, "score": score}
        )
        return score

    recall_service._hybrid_score = wrapped  # type: ignore[assignment]

    def uninstall() -> None:
        recall_service._hybrid_score = orig  # type: ignore[assignment]

    return uninstall


def format_hybrid_line(record: dict[str, Any]) -> str:
    eid = record.get("event_id", "?")
    sim = record.get("cosine_sim", 0.0)
    score = record.get("score", 0.0)
    return f"- {eid}: cosine={sim:.4f} → score={score:.4f}"


# Submodules are imported *after* the public symbols above are defined so that
# ``probes`` 等子模块在 import 阶段可以安全地 ``from . import ProbeResult``。
from . import live_common as live_common  # noqa: E402, F401
from . import probes as probes  # noqa: E402, F401

__all__ = [
    "InstrumentationStore",
    "ProbeResult",
    "WPReporter",
    "default_report_path",
    "install_llm_tracer",
    "install_vector_tracer",
    "install_recall_scoring_tracer",
    "format_hybrid_line",
    "live_common",
    "probes",
]
