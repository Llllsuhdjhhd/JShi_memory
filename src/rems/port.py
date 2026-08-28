"""MemoryBackendPort：匠石记忆后端的稳定接口（对齐匠石《记忆层契约》）。

本模块只定义端口形状与数据模型，不实现引擎逻辑；实现见 REMSPipeline /
MetabolismService / EventService 等（详见 design/210 输入协议）。

硬约束：
- 无抽象事件（不做频繁子集挖掘、ASF、抽象合成）；
- 不建角色系统（对象一律用 01 的 object_id）；
- 只做记忆，不决定主体面；原文保全。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class MemoryExperience(BaseModel):
    """一段经历：统一所有记忆类型的载体。

    - ``text`` 是原文，不归一化、不改写、不切分（输入层）；内部代谢切分是后端内部流程；
    - ``objects`` 为对象映射表（名字 / 称呼 → 01 object_id），覆盖一段经历涉及的对象，
      与是否说话、是否在文本中出现都无关；可为空（= 主体记忆）；
    - ``source_ids`` 来源链（事实 / 活动 / 认知 id），必带，保证来源可追溯；
    - ``origin`` 简单标签：external | internal | dream（字符串，保留扩展能力）。
    """

    subject_id: str = Field(..., description="归属匠石，恒在；绝不放入 object_id")
    text: str = Field(..., description="经历原文（含名字；多轮可整段进入）")
    objects: dict[str, str] = Field(
        default_factory=dict,
        description="对象映射表：名字/称呼 → 01 object_id（含没说话的、未在文本中出现的）",
    )
    source_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="来源链：事实 / 活动 / 认知 id，必带"
    )
    occurred_at: datetime | None = Field(
        default=None, description="经历发生时间；空 = 摄入时刻"
    )
    segment_id: str | None = Field(
        default=None, description="30 侧经历段 id，用于 stored_marks 回填"
    )
    origin: str = Field(default="external", description="external | internal | dream")

    sub_segments: list[SubSegment] = Field(
        default_factory=list,
        description="整批合并时的段级记录（segment_id + 原文 + 对象映射），"
        "供封存事件按内容归属对象，避免整批对象并集串扰。",
    )


class SubSegment(BaseModel):
    """合并批次中的一段原始经历（用于事件→段归属，进而确定事件对象）。"""

    segment_id: str
    text: str
    objects: dict[str, str] = Field(default_factory=dict)


class MemoryBatch(BaseModel):
    """一次写入请求：一条或多条经历（有序：对话 / 经历先后顺序）。"""

    subject_id: str = Field(..., description="实例作用域（恒为当前匠石实例）")
    experiences: tuple[MemoryExperience, ...] = Field(default_factory=tuple)


class BackendIngestResult(BaseModel):
    """记忆写入结果（记忆后端 → JShi / 30）。"""

    subject_id: str
    stored_marks: dict[str, list[str]] = Field(
        default_factory=dict,
        description="segment_id → 本轮封存事件 id 列表（封存几个填几个；30 推进游标与台账用）",
    )
    sealed_event_ids: list[str] = Field(default_factory=list)
    role_ids: list[str] = Field(
        default_factory=list, description="本轮登记 / 更新的 01 对象，可为空"
    )
    unclosed_count: int = 0
    errors: list[str] = Field(default_factory=list, description="非致命错误")


class RecalledFragment(BaseModel):
    """回忆返回的单条记忆（recall 预留）。"""

    event_id: str
    text: str = Field(..., description="原文（content_raw）")
    content: str = Field(
        default="",
        description="按 summary_level 选出的摘要文本，供预算 / 组装（design/1010 §8.2）",
    )
    kind: str | None = None
    object_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    summary_level: str | None = None


@runtime_checkable
class MemoryBackendPort(Protocol):
    """匠石记忆后端四步端口。

    - ``ingest_batch``：记忆写入（本轮实现）；
    - ``recall``：回忆读取（预留，只读；保留记忆恢复，不新增记录、不登记 recall_log）；
    - ``consolidate`` / ``dream``：后台接口（本轮占位）。
    """

    def ingest_batch(self, batch: MemoryBatch) -> BackendIngestResult: ...

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        object_id: str | None = None,
        level: int = 1,
        limit: int | None = None,
        anchor_event_ids: tuple[str, ...] = (),
    ) -> tuple[RecalledFragment, ...]: ...

    def consolidate(
        self,
        subject_id: str,
        *,
        budget: int | None = None,
        cursor: str | None = None,
    ) -> None: ...

    def dream(
        self,
        subject_id: str,
        *,
        object_id: str | None = None,
        budget: int | None = None,
    ) -> None: ...


__all__ = [
    "BackendIngestResult",
    "MemoryBackendPort",
    "MemoryBatch",
    "MemoryExperience",
    "RecalledFragment",
]
