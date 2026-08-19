from __future__ import annotations

import logging

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import RECALL_BLOCK_RELEVANCE_SYSTEM, RECALL_BLOCK_RELEVANCE_USER
from ..models.metabolism import RecallBlock

logger = logging.getLogger(__name__)


class RecallBlockRelevanceSkill:
    """LLM judges how many recalled items relate to current input (rate **b**)."""

    _SNIP = 480

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def evaluate_relevance_ratio(self, current_input: str, block: RecallBlock) -> float:
        """Return related_count / len(items), in ``[0, 1]``."""

        if not block.items:
            return 1.0

        ids: list[str] = []
        chunks: list[str] = []
        for it in block.items:
            text = it.content.strip().replace("\n", " ")
            if len(text) > self._SNIP:
                text = text[: self._SNIP] + "…"
            ids.append(it.event_id)
            chunks.append(f"- event_id `{it.event_id}`\n  {text}")
        recall_sections = "\n\n".join(chunks)

        expected = frozenset(ids)

        user_msg = RECALL_BLOCK_RELEVANCE_USER.format(
            current_input=current_input.strip() or "(空)",
            recall_sections=recall_sections,
        )
        try:
            data = self._llm.complete_json(
                "recall_block_relevance",
                [
                    {"role": "system", "content": RECALL_BLOCK_RELEVANCE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
        except Exception as exc:
            logger.warning("recall block relevance LLM failure: %s", exc)
            return 0.5

        rel = frozenset(str(x) for x in (data.get("related_event_ids") or []) if x)
        unr = frozenset(str(x) for x in (data.get("unrelated_event_ids") or []) if x)

        if rel - expected or unr - expected or rel & unr or (rel | unr) != expected:
            logger.warning(
                "recall block relevance invalid partition (expected=%d rel=%d unr=%d)",
                len(expected),
                len(rel),
                len(unr),
            )
            return 0.5

        return len(rel) / max(1, len(expected))
