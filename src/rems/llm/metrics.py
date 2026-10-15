"""Per-invocation metrics for :class:`LLMProvider` chat completions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMInvocationMetrics:
    """One Chat Completions round-trip as observed by the client.

    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` mirror the API
    ``usage`` object when the server returns it; otherwise they are ``None``.
    """

    task_type: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # 服务端响应里返回的 model 字段（别名解析后的真实模型名）
    response_model: str | None = None
    cached_tokens: int | None = None
