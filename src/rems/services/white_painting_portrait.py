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
        """Refresh once pending white paintings reach the char (or optional count/time) trigger."""
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
        pending_chars = sum(len(item.text) for item in pending)
        chars_due = (
            self._config.portrait_white_painting_trigger_chars > 0
            and pending_chars >= self._config.portrait_white_painting_trigger_chars
        )
        count_due = (
            self._config.portrait_white_painting_trigger_count > 0
            and len(pending) >= self._config.portrait_white_painting_trigger_count
        )
        hours = self._config.portrait_white_painting_trigger_hours
        baseline = portrait.updated_at if portrait is not None else pending[0].created_at
        time_due = hours > 0 and self._clock() - baseline >= timedelta(hours=hours)
        trigger = (
            "manual" if force
            else "chars" if chars_due
            else "count" if count_due
            else "time" if time_due
            else None
        )
        if trigger is None:
            return status

        previous = (
            portrait.longest_level.text
            if portrait is not None and portrait.longest_level is not None else ""
        )
        max_chars = self._config.portrait_max_chars
        fuse = self._config.summary_fuse_min_chars
        max_levels = self._config.portrait_max_levels
        source = [
            {"event_id": item.source_event_id, "white_painting": item.text}
            for item in pending
        ]
        data = self._llm.complete_json("person_portrait", [
            {"role": "system", "content": (
                "根据已有肖像和新增白描整理人物肖像，输出同一肖像的多级分辨率。\n"
                "L1 是最完整的人物描述：保留有依据的人物信息、关系和变化，写成通顺的叙述，"
                "不要写成相处建议，不要虚构。已有肖像中未被新白描推翻的内容保留。\n"
                "L2 起每一级只压缩上一级，约为上一级的一半；删次要句子，人名、数字和专名保持原样，"
                "不从白描里找回上一级已删去的细节。\n"
                f"某一级不超过 {fuse} 字就停止，不再写下一级；最多 {max_levels} 级。\n"
                "只输出一个 JSON 对象：{\"levels\": {\"L1\": \"…\", \"L2\": \"…\"}}。"
            )},
            {"role": "user", "content": (
                f"已有肖像：{previous or '无'}\n"
                f"新增白描：{json.dumps(source, ensure_ascii=False)}\n"
                f"L1 最多 {max_chars} 字。"
            )},
        ])
        raw_levels = data.get("levels") if isinstance(data, dict) else None
        if not isinstance(raw_levels, dict):
            return status
        l1 = str(raw_levels.get("L1") or "").strip()
        if not l1 or len(l1) > max_chars:
            return status

        levels = {"L1": PortraitLevelInfo(
            level="L1", text=l1, budget=max_chars, actual_len=len(l1)
        )}
        current = l1
        for number in range(2, max_levels + 1):
            if len(current) <= fuse:
                break
            shorter = str(raw_levels.get(f"L{number}") or "").strip()
            if not shorter or len(shorter) >= len(current):
                break
            name = f"L{number}"
            budget = max(1, int(len(current) / self._config.portrait_growth))
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
