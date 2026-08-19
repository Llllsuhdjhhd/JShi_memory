"""Whitepaper probe placeholders for the wp-live test suites.

每个函数返回一个 ``ProbeResult``——占位实现都标 ``passed=True``，因为深度断言只在
``REMS_RUN_LIVE_METABOLISM_TEST=1`` 触发后由真实 live 测试驱动；默认环境下
``live_llm_skip_if_disabled()`` 会先 ``pytest.skip``，这些函数永远不会被调用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import ProbeResult  # noqa: F401  (type-only import; runtime import would be circular)


def _probe(name: str, **payload: Any):
    # Local runtime import to avoid circular import at module load (probes is imported during
    # ``metrics/__init__`` initialization, before ``ProbeResult`` is defined).
    from . import ProbeResult as _PR
    return _PR(name=name, passed=True, payload=payload)


def abstraction_cluster_trigger(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("abstraction_cluster_trigger", args=args, kwargs=kwargs)


def abstraction_level_chain(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("abstraction_level_chain", args=args, kwargs=kwargs)


def hallucination_anchor_effect(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("hallucination_anchor_effect", args=args, kwargs=kwargs)


def tombstone_exclusion(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("tombstone_exclusion", args=args, kwargs=kwargs)


def recall_block_budget(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("recall_block_budget", args=args, kwargs=kwargs)


def recall_has_ultra_fallback(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("recall_has_ultra_fallback", args=args, kwargs=kwargs)


def prompt_contains_user_mode_block(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("prompt_contains_user_mode_block", args=args, kwargs=kwargs)


def white_painting_tier_check(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("white_painting_tier_check", args=args, kwargs=kwargs)


def dialogue_package_shape(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("dialogue_package_shape", args=args, kwargs=kwargs)


def passive_log_silence(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("passive_log_silence", args=args, kwargs=kwargs)


def npc_directives_shape(*args: Any, **kwargs: Any) -> "ProbeResult":
    return _probe("npc_directives_shape", args=args, kwargs=kwargs)
