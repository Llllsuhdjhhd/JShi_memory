from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ObjectMemoryEntry(BaseModel):
    """对象记忆时间线条目：某对象在一事件中的一句摘要快照。

    无情感（情感只属于主体，见 design/410/810）；``name`` 仅用于呈现
    （与输入映射表一致，便于阅读，如回忆时）。
    """

    subject_id: str = ""
    object_id: str
    name: str | None = None
    event_id: str
    summary: str
    create_time: datetime = Field(default_factory=datetime.now)


__all__ = ["ObjectMemoryEntry"]
