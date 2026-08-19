# 对外导出的 Pydantic 领域模型（事件、角色、代谢/回忆上下文）。

from .event import (
    BasicEmotionVector,
    Event,
    EventRoleEntry,
    EventStatus,
    Importance,
    EmotionalModel,
    RoleSnapshot,
    generate_event_id,
)
from .role import Role, WhitePaintingEntry, generate_role_id
from .metabolism import (
    Shadow,
    UnclosedEvent,
    RecallItem,
    RecallBlock,
    ContextPackage,
)

__all__ = [
    "Event",
    "EventRoleEntry",
    "EventStatus",
    "Importance",
    "BasicEmotionVector",
    "EmotionalModel",
    "RoleSnapshot",
    "generate_event_id",
    "Role",
    "WhitePaintingEntry",
    "generate_role_id",
    "Shadow",
    "UnclosedEvent",
    "RecallItem",
    "RecallBlock",
    "ContextPackage",
]
