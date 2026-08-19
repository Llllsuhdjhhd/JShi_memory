from __future__ import annotations

import logging

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import BOUNDARY_SYSTEM, BOUNDARY_USER
from ..models.metabolism import UnclosedEvent

logger = logging.getLogger(__name__)

SHADOW_COMPACTION_SYSTEM = """\
你是 REMS 残影压实组件。将支离破碎的未完成叙事片段改写为连贯的背景状态描述。
保留全部事实要点，消除语法孤儿句，不要添加新事实。"""

SHADOW_COMPACTION_USER = """\
## 待压实残影
{content}

返回 JSON：{{"compacted_content": "改写后的连贯文本"}}"""


class ShadowCompactionSkill:
    """LLM shadow compaction (§4.2.2 mechanism 2)."""

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def compact(self, content: str) -> str:
        if not content.strip():
            return content
        try:
            data = self._llm.complete_json(
                "boundary_detection",
                [
                    {"role": "system", "content": SHADOW_COMPACTION_SYSTEM},
                    {"role": "user", "content": SHADOW_COMPACTION_USER.format(content=content)},
                ],
            )
            return str(data.get("compacted_content") or content).strip() or content
        except Exception as exc:
            logger.warning("Shadow compaction failed: %s", exc)
            return content

    def should_compact(self, unclosed_events: list[UnclosedEvent]) -> bool:
        threshold = self._config.lifecycle.shadow_compaction_fragment_threshold
        total_frags = sum(len(ue.content_fragments or []) for ue in unclosed_events)
        return total_frags >= threshold
