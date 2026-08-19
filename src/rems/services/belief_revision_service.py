from __future__ import annotations

import logging
from typing import Optional

# 动态信念修正：自然演进依赖白描时序+遗忘；技术性错误用墓碑 tombstone 逻辑覆写（白皮书 4.3）。

from ..config import REMSConfig
from ..models.event import EventStatus
from ..storage.repository import EventRepository, RoleRepository

logger = logging.getLogger(__name__)


class BeliefRevisionService:
    """Handles knowledge-conflict resolution (Dynamic Belief Revision; white paper section 4.3).

    Two conflict archetypes:

    1. **Natural evolution** — facts change over time (e.g. allergy that
       developed later).  Handled implicitly: the white-painting timeline's
       chronological ordering plus the AE-weighted forgetting mechanism ensures
       older entries naturally decay.  No explicit action needed.

    2. **Technical / generational correction (Tombstone)** — a newer or more
       capable inference conclusively supersedes a prior one (e.g. a corrected
       diagnosis, a revised factual claim).  The old event is logically
       overwritten (tombstoned); it is retained in the DB for audit but
       excluded from all future retrieval paths.

    知识冲突的两类底层逻辑（白皮书 4.3）：

    1. **事实的自然演进**：偏好、过敏等随时间真实变化，并非模型错误。系统依赖白描时间序、
       高 AE 条目的抗遗忘与低 AE 条目的衰减，使旧状态自然让位于新状态，通常无需显式删除。

    2. **技术性/代差修正（墓碑 Tombstone）**：新版本模型或更可靠证据推翻旧结论时，对旧事件
       执行逻辑覆写：标记 ``is_tombstoned``、状态静默、在 ``insight`` 追加审计说明；库中仍保留行记录，
       但向量检索与回忆路径必须排除墓碑事件，避免错误事实再次进入上下文。
    """

    def __init__(self, config: REMSConfig, event_repo: EventRepository, role_repo: RoleRepository):
        self._config = config
        self._event_repo = event_repo
        self._role_repo = role_repo

    # ------------------------------------------------------------------
    # Tombstone API
    # ------------------------------------------------------------------

    def tombstone_event(
        self,
        event_id: str,
        *,
        reason: str,
        replacement_event_id: Optional[str] = None,
    ) -> bool:
        """Mark *event_id* as tombstoned (logically overwritten).

        The event is silenced in retrieval but preserved for audit.  An
        optional *replacement_event_id* links to the correcting event.

        Returns True if the event was found and tombstoned.

        将 *event_id* 标记为墓碑：检索侧静默（``SILENT``）、``is_tombstoned=True``，
        并在 ``insight`` 追加带 ``tombstone_prefix`` 的审计说明；可选 *replacement_event_id*
        指向修正后的新事件。若事件不存在返回 False；已墓碑则视为成功并返回 True。
        """
        event = self._event_repo.get(event_id)
        if event is None:
            logger.warning("Tombstone requested for unknown event %s", event_id)
            return False

        if event.is_tombstoned:
            logger.info("Event %s is already tombstoned", event_id)
            return True

        tombstone_note = (
            f"{self._config.tombstone_prefix} {reason}"
            + (f" → replaced by {replacement_event_id}" if replacement_event_id else "")
        )

        self._event_repo.update_status(
            event_id,
            status=EventStatus.SILENT,
            is_tombstoned=True,
        )
        # Append tombstone note to insight for audit trail
        existing_insight = event.insight or ""
        new_insight = (existing_insight + "\n" + tombstone_note).strip()
        # Re-save via full record update
        event.status = EventStatus.SILENT
        event.is_tombstoned = True
        event.insight = new_insight
        self._event_repo.save(event)

        logger.info("Tombstoned event %s: %s", event_id, reason)
        return True

    def revive_event(self, event_id: str) -> bool:
        """Undo a tombstone (e.g. the revision was itself erroneous).

        撤销墓碑：将 ``is_tombstoned`` 置 False 并恢复 ``ACTIVE``，用于误修正或运维回滚。
        不自动清除 ``insight`` 中历史审计文本，需由上层策略决定是否清理。
        """
        event = self._event_repo.get(event_id)
        if event is None:
            return False
        event.is_tombstoned = False
        event.status = EventStatus.ACTIVE
        self._event_repo.save(event)
        logger.info("Revived event %s", event_id)
        return True

    # ------------------------------------------------------------------
    # Conflict detection helper
    # ------------------------------------------------------------------

    def detect_conflicts(self, role_id: str, *, top_n: int = 50) -> list[dict]:
        """Return pairs of potentially conflicting white-painting entries.

        Conflict heuristic: two entries for the same role have near-identical
        L1 topics but opposite valence, which may indicate a factual contradiction worth reviewing.

        This is a lightweight advisory scan — final resolution is by the LLM
        or operator.

        返回同一角色白描时间线中**可能**互斥的条目对（启发式）：若两条目呈现相反效价，
        则标记为待人工或 LLM 复核的冲突线索。
        该扫描仅为建议列表，不构成自动仲裁；与墓碑 API 正交。
        """
        entries = self._role_repo.get_white_painting(role_id, limit=top_n)
        conflicts: list[dict] = []
        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                a_joy = a.emotional_model.valence
                b_joy = b.emotional_model.valence
                # Opposite sentiment polarity may signal contradiction
                if a_joy * b_joy < -0.2:
                    conflicts.append({
                        "event_a": a.event_id,
                        "event_b": b.event_id,
                        "summary_a": a.role_summary[:80],
                        "summary_b": b.role_summary[:80],
                        "sentiment_delta": round(abs(a_joy - b_joy), 3),
                    })
        return conflicts
