from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..config import REMSConfig
from .act_text import clamp_embed_text, resolve_query_act_text

if TYPE_CHECKING:
    from ..skills.recall_query_compress import RecallQueryCompressSkill

logger = logging.getLogger(__name__)

QueryActSource = Literal["passthrough", "llm_compress", "truncate_fallback"]


@dataclass(frozen=True)
class QueryActResult:
    act_text: str
    source: QueryActSource


class QueryActProcessor(Protocol):
    def resolve_act_text(self, text: str) -> QueryActResult: ...


class PassthroughQueryActProcessor:
    """Return text unchanged (after strip)."""

    def resolve_act_text(self, text: str) -> QueryActResult:
        cleaned = (text or "").strip()
        return QueryActResult(act_text=cleaned, source="passthrough")


class TruncateFallbackQueryActProcessor:
    """Legacy mechanical truncate when LLM compress is off or fails."""

    def __init__(self, config: REMSConfig):
        self._config = config

    def resolve_act_text(self, text: str) -> QueryActResult:
        cleaned = (text or "").strip()
        if not cleaned:
            return QueryActResult(act_text="", source="passthrough")
        tb = self._config.tri_band
        if len(cleaned) <= tb.query_act_max_chars:
            return QueryActResult(
                act_text=clamp_embed_text(cleaned, tb.act_embed_max_chars),
                source="passthrough",
            )
        truncated = resolve_query_act_text(cleaned, tb)
        return QueryActResult(act_text=truncated, source="truncate_fallback")


class LLMQueryActProcessor:
    """Compress long query text via RecallQueryCompressSkill before act embed."""

    def __init__(self, skill: RecallQueryCompressSkill, config: REMSConfig):
        self._skill = skill
        self._config = config
        self._fallback = TruncateFallbackQueryActProcessor(config)

    def resolve_act_text(self, text: str) -> QueryActResult:
        cleaned = (text or "").strip()
        if not cleaned:
            return QueryActResult(act_text="", source="passthrough")

        min_chars = self._config.recall_query_compress_min_chars
        cap = self._config.tri_band.query_act_compress_max_chars
        if len(cleaned) <= min_chars and len(cleaned) <= cap:
            return QueryActResult(act_text=cleaned, source="passthrough")

        try:
            compressed = self._skill.compress(cleaned)
            if compressed:
                safe = clamp_embed_text(compressed, cap)
                if len(compressed) > cap:
                    logger.warning(
                        "query act compress exceeded cap (%d > %d), safety clamped",
                        len(compressed),
                        cap,
                    )
                return QueryActResult(act_text=safe, source="llm_compress")
        except Exception as exc:
            logger.warning("query act LLM compress failed: %s", exc)

        return self._fallback.resolve_act_text(cleaned)


def build_query_act_processor(
    config: REMSConfig,
    skill: RecallQueryCompressSkill | None = None,
) -> QueryActProcessor:
    if config.recall_query_compress_enabled and skill is not None:
        return LLMQueryActProcessor(skill, config)
    return TruncateFallbackQueryActProcessor(config)
