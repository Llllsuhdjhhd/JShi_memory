from __future__ import annotations

import logging
from typing import Optional

# 角色抽取：从 L0 得到 role_list 候选、快照层级与 8 维情绪；支持已知角色表去代词化（白皮书 2.1）。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import ROLE_EXTRACTION_SYSTEM, ROLE_EXTRACTION_USER, build_user_mode_block
from ..models.event import (
    BasicEmotionVector,
    EmotionalModel,
    EventRoleEntry,
    Importance,
    RoleSnapshot,
)
from ..models.role import Role

logger = logging.getLogger(__name__)


class ExtractedRole(BaseModel):
    role_id: Optional[str] = None
    name: str = ""
    entity_type: str = "person"
    importance: str = "C"
    snapshot: RoleSnapshot = Field(default_factory=RoleSnapshot)
    emotional_model: EmotionalModel = Field(default_factory=EmotionalModel)


class RoleExtractionResult(BaseModel):
    roles: list[ExtractedRole] = Field(default_factory=list)


class RoleExtractionSkill:
    """Extracts structured roles (snapshots + 8-d emotion) from raw event text via LLM JSON.

    将 ``content_raw`` 与可选 ``known_roles`` 描述一并送入 ``role_extraction`` 模型，解析为
    ``ExtractedRole`` 列表；``to_event_role_entry`` 再转为持久化用的 ``EventRoleEntry``（白皮书 2.1，去代词化在 prompt 层强调）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def extract(self, content_raw: str, known_roles: list[Role] | None = None, budget: "CompressionBudget" | None = None) -> RoleExtractionResult:
        known_desc = "无已知角色" if not known_roles else "\n".join(
            f"- {r.role_id}: {r.name} ({r.entity_type}), 别名={r.aliases}"
            for r in (known_roles or [])
        )

        # 指数级衰减预算注入
        snapshot_budgets_text = "按系统默认要求"
        if budget:
            snapshot_budgets_text = "\n".join(f"- {lvl}: {b} 字以内" for lvl, b in budget.snapshot_level_budgets.items())

        user_msg = ROLE_EXTRACTION_USER.format(
            known_roles=known_desc,
            content_raw=content_raw,
            snapshot_budgets=snapshot_budgets_text,
        )

        # 白皮书 2.2：在系统提示词首部注入单人/多人模式块，指导模型做代词消解。
        system_msg = build_user_mode_block(self._config) + ROLE_EXTRACTION_SYSTEM

        data = self._llm.complete_json(
            "role_extraction",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        return parse_role_extraction_data(data)

    @staticmethod
    def to_event_role_entry(er: ExtractedRole, assigned_role_id: str) -> EventRoleEntry:
        return EventRoleEntry(
            role_id=assigned_role_id,
            importance=Importance(er.importance) if er.importance in Importance.__members__ else Importance.C,
            role_snapshot=er.snapshot,
            emotional_model=er.emotional_model,
        )


def parse_role_extraction_data(data: dict) -> RoleExtractionResult:
    """Parse LLM JSON payload into ``RoleExtractionResult`` (shared by skill + ingest session)."""
    extracted: list[ExtractedRole] = []
    for rd in data.get("roles", []):
        snap = rd.get("snapshot") or {}
        emo = rd.get("emotion") or {}
        emotion_init = {}
        if isinstance(emo, dict):
            emotion_init = {
                k: _safe_float(v)
                for k, v in emo.items()
                if k in BasicEmotionVector.model_fields
            }
        emotion = BasicEmotionVector(**emotion_init)

        extracted.append(ExtractedRole(
            role_id=rd.get("role_id"),
            name=rd.get("name", ""),
            entity_type=rd.get("entity_type", "person"),
            importance=rd.get("importance", "C"),
            snapshot=RoleSnapshot(
                l1_mention=snap.get("l1_mention"),
                l2_interaction=snap.get("l2_interaction"),
                l3_decision=snap.get("l3_decision"),
            ),
            emotional_model=EmotionalModel.from_emotion(emotion),
        ))

    return RoleExtractionResult(roles=extracted)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
