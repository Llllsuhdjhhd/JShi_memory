from __future__ import annotations

import logging
import random

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..models.event import Event, EventStatus, generate_event_id
from ..storage.repository import EventRepository

logger = logging.getLogger(__name__)

DREAM_SYSTEM = """\
你是 REMS 离线梦境演化组件。从若干历史事件中梳理潜在逻辑关系。
若发现有效模式，输出压缩 content_raw；否则 cognitive_relation 为 NONE。"""


class DreamEvolutionSkill:
    """Offline dream consolidation skill (§4.7.2)."""

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def synthesize_dream(self, events: list[Event]) -> Event | None:
        if not events:
            return None
        body = "\n\n".join(f"### {e.event_id}\n{e.content_raw}" for e in events)
        try:
            data = self._llm.complete_json(
                "abstraction",
                [
                    {"role": "system", "content": DREAM_SYSTEM},
                    {"role": "user", "content": body + '\n\n返回 JSON：{"cognitive_relation":"...", "content_raw":"..."}'},
                ],
            )
        except Exception as exc:
            logger.warning("Dream synthesis failed: %s", exc)
            return None
        if str(data.get("cognitive_relation", "NONE")).upper() == "NONE":
            return None
        return Event(
            event_id=generate_event_id(),
            content_raw=data.get("content_raw", ""),
            is_abstract=True,
            status=EventStatus.ACTIVE,
            source_events=[e.event_id for e in events],
            role_list=[],
            origin="dream",
        )


def run_dream_consolidation(
    config: REMSConfig,
    llm: LLMProvider,
    event_repo: EventRepository,
    batch_size: int | None = None,
) -> list[Event]:
    if not config.lifecycle.dream_enabled:
        return []
    basics = event_repo.list_all(is_abstract=False)
    if len(basics) < 2:
        return []
    k = batch_size or config.lifecycle.dream_batch_size
    sample = random.sample(basics, min(k, len(basics)))
    skill = DreamEvolutionSkill(llm, config)
    ev = skill.synthesize_dream(sample)
    return [ev] if ev else []
