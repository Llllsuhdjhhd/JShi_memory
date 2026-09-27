"""Explicit, batch-triggered descriptive portrait refresh from stored white paintings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from typing import Callable

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..models.portrait import ObjectPortrait, PendingPortraitSummary, PortraitLevelInfo
from ..storage.repository import PortraitRepository, RoleRepository


@dataclass(frozen=True)
class WhitePaintingPortraitSyncResult:
    object_id: str
    imported_white_paintings: int
    pending_white_paintings: int
    updated: bool
    trigger: str | None = None
    levels: tuple[str, ...] = ()


class WhitePaintingPortraitBuilder:
    """Refresh the descriptive portrait without running in Event ingest/record."""

    def __init__(
        self,
        config: REMSConfig,
        white_paintings: RoleRepository,
        portraits: PortraitRepository,
        llm: LLMProvider | None,
        *,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._config = config
        self._white_paintings = white_paintings
        self._portraits = portraits
        self._llm = llm
        self._clock = clock

    def refresh_due(
        self, subject_id: str
    ) -> tuple[WhitePaintingPortraitSyncResult, ...]:
        """Periodic batch entry point; each person is checked against both triggers."""
        if not subject_id.strip():
            raise ValueError("subject_id is required")
        return tuple(
            self.refresh(subject_id, object_id)
            for object_id in self._white_paintings.list_white_painting_object_ids(subject_id)
        )

    def refresh(
        self, subject_id: str, object_id: str, *, force: bool = False
    ) -> WhitePaintingPortraitSyncResult:
        """Call from a periodic job; count can also make a call immediately due."""
        if not subject_id.strip() or not object_id.strip():
            raise ValueError("subject_id and object_id are required")
        portrait = self._portraits.get_portrait(object_id)
        if portrait is not None and portrait.subject_id != subject_id:
            raise ValueError("portrait belongs to a different subject")
        summaries = self._portraits.list_summaries(object_id)
        if any(item.subject_id != subject_id for item in summaries):
            raise ValueError("portrait summaries belong to a different subject")

        prefix = f"wp:{subject_id}:{object_id}:"
        known_events = {
            item.source_event_id for item in summaries
            if item.summary_id.startswith(prefix) and item.source_event_id
        }
        imported = 0
        for painting in self._white_paintings.get_white_painting(object_id):
            if (
                painting.subject_id != subject_id
                or not painting.event_id
                or not painting.role_summary.strip()
                or painting.event_id in known_events
            ):
                continue
            self._portraits.add_summary(PendingPortraitSummary(
                summary_id=f"{prefix}{painting.event_id}",
                subject_id=subject_id,
                object_id=object_id,
                text=painting.role_summary.strip(),
                source_event_id=painting.event_id,
                weight=max(0.0, min(1.0, painting.memory_weight)),
                created_at=painting.create_time,
            ))
            known_events.add(painting.event_id)
            imported += 1

        summaries = self._portraits.list_summaries(object_id)
        pending = tuple(
            item for item in summaries
            if item.summary_id.startswith(prefix) and not item.incorporated
        )
        status = WhitePaintingPortraitSyncResult(
            object_id, imported, len(pending), False
        )
        if not pending or self._llm is None:
            return status
        count_due = (
            self._config.portrait_white_painting_trigger_count > 0
            and len(pending) >= self._config.portrait_white_painting_trigger_count
        )
        hours = self._config.portrait_white_painting_trigger_hours
        baseline = portrait.updated_at if portrait is not None else pending[0].created_at
        time_due = hours > 0 and self._clock() - baseline >= timedelta(hours=hours)
        trigger = "manual" if force else "count" if count_due else "time" if time_due else None
        if trigger is None:
            return status

        previous = (
            portrait.longest_level.text
            if portrait is not None and portrait.longest_level is not None else ""
        )
        max_chars = self._config.portrait_max_chars
        source = [
            {"event_id": item.source_event_id, "white_painting": item.text}
            for item in pending
        ]
        first = self._llm.complete_json("person_portrait", [
            {"role": "system", "content": (
                "根据已有白描整理人物肖像。L1 是最完整的人物描述；保留有依据的人物信息、"
                "关系和变化，整理语句，不要写成相处建议，不要虚构。"
                "输出 JSON：{\"text\":\"人物肖像\"}。"
            )},
            {"role": "user", "content": (
                f"已有肖像：{previous or '无'}\n"
                f"新增白描：{json.dumps(source, ensure_ascii=False)}\n"
                f"L1 最多 {max_chars} 字。"
            )},
        ])
        l1 = str(first.get("text") or "").strip()
        if not l1 or len(l1) > max_chars:
            return status

        levels = {"L1": PortraitLevelInfo(
            level="L1", text=l1, budget=max_chars, actual_len=len(l1)
        )}
        current = l1
        for number in range(2, self._config.portrait_max_levels + 1):
            if len(current) <= self._config.summary_fuse_min_chars:
                break
            budget = max(1, int(len(current) / self._config.portrait_growth))
            data = self._llm.complete_json("person_portrait", [
                {"role": "system", "content": (
                    "将上一层人物肖像继续保真压缩。保留最重要的人物信息，"
                    "不要加入新事实或相处建议。输出 JSON：{\"text\":\"压缩文本\"}。"
                )},
                {"role": "user", "content": (
                    f"上一层：{current}\nL{number} 最多 {budget} 字。"
                )},
            ])
            shorter = str(data.get("text") or "").strip()
            if not shorter or len(shorter) > budget or len(shorter) >= len(current):
                break
            name = f"L{number}"
            levels[name] = PortraitLevelInfo(
                level=name, text=shorter, budget=budget, actual_len=len(shorter)
            )
            current = shorter

        source_chars = (portrait.total_content_len if portrait else 0) + sum(
            len(item.text) for item in pending
        )
        now = self._clock()
        self._portraits.save_portrait_and_mark(ObjectPortrait(
            subject_id=subject_id,
            object_id=object_id,
            name=portrait.name if portrait else None,
            levels=levels,
            max_level="L1",
            total_content_len=source_chars,
            compression_ratio=round(source_chars / len(l1), 4),
            fatigue=portrait.fatigue if portrait else self._config.portrait_fatigue,
            created_at=portrait.created_at if portrait else now,
            updated_at=now,
        ), [item.summary_id for item in pending])
        return WhitePaintingPortraitSyncResult(
            object_id, imported, 0, True, trigger, tuple(levels)
        )
