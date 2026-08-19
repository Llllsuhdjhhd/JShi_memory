from __future__ import annotations

import logging

from ..config import REMSConfig
from ..embedding.act_text import clamp_embed_text, query_act_compress_bounds
from ..llm.provider import LLMProvider
from ..llm.prompts import (
    RECALL_QUERY_COMPRESS_FOLLOWUP,
    RECALL_QUERY_COMPRESS_RETRY_NOTE,
    RECALL_QUERY_COMPRESS_SYSTEM,
    RECALL_QUERY_COMPRESS_USER,
    RECALL_QUERY_COMPRESS_WRITING_GUIDE,
)

logger = logging.getLogger(__name__)


def _format_writing_guide(config: REMSConfig, text_len: int) -> str:
    cap, target, min_chars = query_act_compress_bounds(text_len, config.tri_band)
    return RECALL_QUERY_COMPRESS_WRITING_GUIDE.format(
        source_chars=text_len,
        max_chars=cap,
        target_chars=target,
        min_chars=min_chars,
    )


def format_compress_followup(
    config: REMSConfig,
    text_len: int,
    *,
    retry_note: str = "",
) -> str:
    return RECALL_QUERY_COMPRESS_FOLLOWUP.format(
        writing_guide=_format_writing_guide(config, text_len),
        retry_note=retry_note,
    )


def compress_act_query_from_llm(
    llm: LLMProvider,
    config: REMSConfig,
    messages: list[dict[str, str]],
    text_len: int,
    *,
    model: str | None = None,
) -> str:
    """Call LLM compress; retry once if output falls below min_chars."""
    cap, target, min_chars = query_act_compress_bounds(text_len, config.tri_band)

    def _call(msgs: list[dict[str, str]]) -> str:
        data = llm.complete_json(
            "recall_query_compress", msgs, temperature=0.1, model=model,
        )
        act_query = str(data.get("act_query") or "").strip()
        if not act_query:
            raise ValueError("empty act_query from LLM")
        if len(act_query) > cap:
            logger.warning(
                "query act compress exceeded cap (%d > %d), clamping",
                len(act_query),
                cap,
            )
        return clamp_embed_text(act_query, cap)

    act_query = _call(messages)
    if len(act_query) >= min_chars:
        return act_query

    logger.warning(
        "query act compress too short (%d < min_chars=%d), retrying",
        len(act_query),
        min_chars,
    )
    retry_note = RECALL_QUERY_COMPRESS_RETRY_NOTE.format(
        prev_chars=len(act_query),
        min_chars=min_chars,
        target_chars=target,
    )
    retry_user = format_compress_followup(config, text_len, retry_note=retry_note)
    retry_messages = list(messages[:-1]) + [{"role": "user", "content": retry_user}]
    act_query = _call(retry_messages)
    if len(act_query) < min_chars:
        logger.warning(
            "query act compress still short after retry (%d < %d)",
            len(act_query),
            min_chars,
        )
    return act_query


class RecallQueryCompressSkill:
    """LLM-compress long recall query text into an act-band retrieval summary."""

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def compress(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        user_msg = RECALL_QUERY_COMPRESS_USER.format(
            content=text,
            writing_guide=_format_writing_guide(self._config, len(text)),
        )
        messages = [
            {"role": "system", "content": RECALL_QUERY_COMPRESS_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        return compress_act_query_from_llm(
            self._llm, self._config, messages, len(text),
        )
