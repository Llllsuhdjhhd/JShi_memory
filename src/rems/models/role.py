from __future__ import annotations

import secrets
import time
from datetime import datetime

from pydantic import BaseModel, Field

from .event import EmotionalModel, Importance

# 角色系统模型：全局 role_id 与时间序白描（White-painting）。
# 见《REMS 记忆系统规范解析》第 2 章（白描与遗忘）。语义卡片已废除。


import uuid

def generate_role_id() -> str:
    return f"ROL-{uuid.uuid4().hex}"


class WhitePaintingEntry(BaseModel):
    """Single node on a role's chronological white-painting timeline.

    角色白描（时间序）上的单个节点：每当某基本事件引用该 ``role_id``，将对应瞬时记录追加到轴上，
    按 ``create_time`` 升序浏览即得行为与情感演变轨迹（白皮书 2.2）。``memory_weight`` 承载 arousal 映射后的抗遗忘强度。
    """

    event_id: str  # 来源基本事件，便于溯源。
    role_summary: str  # 该角色在此事件中的白描摘要（多取 L2 互动层）。
    emotional_model: EmotionalModel = Field(default_factory=EmotionalModel)
    importance: Importance = Importance.C
    create_time: datetime = Field(default_factory=datetime.now)
    # 由 arousal 映射的记忆权重 [0,1]，越高越抗遗忘（白皮书 2.2 动态遗忘）。
    memory_weight: float = 0.0
    # 动态遗忘因子，基于情绪极端值初始化，低于 0.02 则静默。
    forgetting_factor: float = 1.0
    # 基础遗忘因子，记录特殊事件的高额初始值（如初恋 = 100）。
    base_forgetting_factor: float = 1.0
    # 上次计算/访问时间，用于动态遗忘因子衰减
    last_accessed_time: datetime = Field(default_factory=datetime.now)
    # 该条目是否包含可疑标记（事件边界/仲裁不确定的产物）
    is_suspicious: bool = False


class Role(BaseModel):
    # role_id 为持久容器指针；事件内应去代词化并挂接此 ID（白皮书 2.1）。
    role_id: str = Field(default_factory=generate_role_id)
    name: str
    entity_type: str = "person"
    aliases: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    white_painting: list[WhitePaintingEntry] = Field(default_factory=list)  # 时间序事实流（2.2）。
    # 角色在系统中是否属于强制生成的“可疑记录”（身份不确切等）
    is_suspicious: bool = False
