from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import TriBandConfig
    from ..models.event import Event


def clamp_embed_text(text: str, max_chars: int) -> str:
    """Hard cap for embed input; only used when text still exceeds model-safe limit."""
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _summary_level_keys(event: Event) -> list[str]:
    summaries = event.summaries or {}
    keys: list[str] = []
    for key in summaries:
        if key.startswith("L") and key[1:].isdigit():
            keys.append(key)
    return sorted(keys, key=lambda k: int(k[1:]))


def pick_summary(event: Event, level: str) -> str | None:
    """Return summary text for *level* if present and non-empty."""
    text = (event.summaries or {}).get(level)
    if text and str(text).strip():
        return str(text).strip()
    return None


def resolve_event_act_text(event: Event, tb: TriBandConfig) -> str:
    """Choose act-band embed text: max information within ``act_embed_max_chars``."""
    raw = (event.content_raw or "").strip()
    raw_len = len(raw)
    cap = tb.act_embed_max_chars

    if raw_len <= tb.act_short_raw_chars:
        preferred_levels = [tb.act_short_level]
    else:
        preferred_levels = [tb.act_long_level, "L1"]

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(text: str | None) -> None:
        if text and text not in seen:
            candidates.append(text)
            seen.add(text)

    for lvl in preferred_levels:
        _add(pick_summary(event, lvl))
    for lvl in _summary_level_keys(event):
        if lvl not in preferred_levels:
            _add(pick_summary(event, lvl))

    summary_candidates = list(candidates)
    fitting = [c for c in summary_candidates if len(c) <= cap]
    if fitting:
        return max(fitting, key=len)

    if raw:
        _add(raw)

    fitting = [c for c in candidates if len(c) <= cap]
    if fitting:
        return max(fitting, key=len)

    # None fit: prefer coarsest (highest L) shortest summary, then safety clamp.
    level_keys = _summary_level_keys(event)
    coarse_texts = [
        pick_summary(event, lvl)
        for lvl in reversed(level_keys)
        if pick_summary(event, lvl)
    ]
    if coarse_texts:
        shortest = min(coarse_texts, key=len)
        return clamp_embed_text(shortest, cap)

    if raw:
        return clamp_embed_text(raw, cap)
    return event.event_id


def resolve_query_act_text(text: str, tb: TriBandConfig) -> str:
    """Legacy truncate for query act when LLM compress is disabled or fails."""
    text = (text or "").strip()
    cap = tb.query_act_max_chars
    if cap <= 0 or len(text) <= cap:
        return text
    if tb.query_act_use_tail:
        return text[-cap:]
    return text[:cap]


def query_act_compress_bounds(text_len: int, tb: TriBandConfig) -> tuple[int, int, int]:
    """Return ``(max_chars, target_chars, min_chars)`` for recall query LLM compress."""
    cap = tb.query_act_compress_max_chars
    target = min(cap, tb.query_act_max_chars)
    # 下限 = target；若原文短于 target 则取原文长度（无法再扩写）
    min_chars = min(target, text_len) if text_len > 0 else 0
    return cap, target, min_chars
