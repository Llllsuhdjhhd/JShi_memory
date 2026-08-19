"""White-paper live-report support package.

本子包为 ``tests/test_wp_live_*.py`` 提供三类能力：

* :mod:`tests.metrics.reporter` — ``WPReporter`` 渲染 Markdown 指标报告并刷新总索引 ``wp_live_INDEX.md``。
* :mod:`tests.metrics.instrumentation` — 在 Live 测试期间拦截 LLM 调用、向量检索与回忆混合打分中间项，
  将样本收集到 ``InstrumentationStore``，供报告章节引用。
* :mod:`tests.metrics.probes` — 对照《REMS 记忆系统规范解析》各章节的指标探针函数，返回标准 ``ProbeResult`` 结构。
"""

from .instrumentation import (
    InstrumentationStore,
    format_hybrid_line,
    install_llm_tracer,
    install_recall_scoring_tracer,
    install_vector_tracer,
)
from .probes import ProbeResult
from .reporter import SectionCtx, WPReporter, default_report_path, report_dir_from_env

__all__ = [
    "InstrumentationStore",
    "ProbeResult",
    "SectionCtx",
    "WPReporter",
    "default_report_path",
    "format_hybrid_line",
    "install_llm_tracer",
    "install_recall_scoring_tracer",
    "install_vector_tracer",
    "report_dir_from_env",
]
