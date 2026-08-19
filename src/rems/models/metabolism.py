from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# 代谢中间态与回忆上下文包：残影 Shadow、未完成事件库条目、RecallBlock、ContextPackage。
# 见《REMS 记忆系统规范解析》第 4.1（残影与剥离）、4.4（回忆与上下文组装）、第 5 章多场景输入。


class Shadow(BaseModel):
    """Raw-text buffer derived from unclosed events.

    残影：由所有未完成事件的原始片段拼接而成，作为当前摄入的即时上下文（白皮书 4.1）。
    在代谢过程中，它作为边界检测的输入之一，并在代谢后根据最新的未完成事件库进行同步更新。
    """

    content: str = ""
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def length(self) -> int:
        return len(self.content)


class UnclosedEvent(BaseModel):
    """A logically-opened but not-yet-closed event object.

    未完成事件：叙事上已启动但缺关键结果或上下文的事实对象（白皮书 4.1）。
    所有的未完成事件拼接在一起构成了系统的“残影”。当边界模型识别到逻辑闭环时，
    对应的未完成事件将被封存为基本事件，并从残影中移除。

    分裂链路字段（2026-05 新增）：
        ``split_prefix_event_ids`` —— 若本 UC 是边界模型执行 80/20 强制分裂的"尾段"，
        则这里按分裂顺序列出它前面的前缀事件 id（可以跨多轮累积，故允许 len>=1）。
        在 UC 最终闭环为事件时，由 ``MetabolismService`` 把该列表继承到目标事件的
        ``split_prefix_event_ids`` 上；在回忆时由 ``RecallService`` 硬性把这些前缀事件
        拉进回忆块（允许降档压缩但不丢弃）。
        ``oversized`` —— 评估器判定该 UC 超过 force_threshold 且修复失败时置为 True，
        只作为审计标记；不会触发强制封存（封存的兜底仅保留在物理红线一层）。
    """

    id: str
    content_fragments: list[str] = Field(default_factory=list)
    identified_roles: list[str] = Field(default_factory=list)
    logical_gaps: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_hit_time: datetime = Field(default_factory=datetime.now)

    # 分裂链路：前缀事件 id（按从远到近的顺序）。
    split_prefix_event_ids: list[str] = Field(default_factory=list)
    # 审计标记：模型失败切分后残留的超长未完成条目。
    oversized: bool = False

    @property
    def total_length(self) -> int:
        return sum(len(f) for f in self.content_fragments)

    @property
    def merged_content(self) -> str:
        return "\n".join(self.content_fragments)


class RecallItem(BaseModel):
    """One recalled snippet placed into the recall block."""

    """单条回忆条目：``event_id`` 为真实事件 ID；
    ``content`` 为已按预算降级后的文本；``score`` 为融合相关度；``summary_level`` 标记选用的摘要层级或 ``ultra``。
    """

    event_id: str
    content: str
    score: float = 0.0
    summary_level: Optional[str] = None


class RecallBlock(BaseModel):
    """Assembled recall context (capped at ~1/6.6 of the configured context window).

    回忆块：由若干 ``RecallItem`` 组成，每条含被选用摘要级别 ``summary_level``（如 L1、card、ultra）；
    ``total_length`` 为拼接后字符数上限约束的监控值（白皮书 4.4，工程上对应 ``physical_redline``）。
    """

    items: list[RecallItem] = Field(default_factory=list)
    total_length: int = 0

    def recompute_length(self) -> int:
        self.total_length = sum(len(it.content) for it in self.items)
        return self.total_length


class ContextPackage(BaseModel):
    """Final context fed to the LLM: recall + shadow + current input.

    代谢素材包：对话场景下由 ``assemble()`` 拼出带 ``[历史回忆]``、``[残影缓冲]``、``[当前输入]`` 标记的大文本，
    作为上游 LLM 的 user/system 素材；被动日志模式可不构建该对象（白皮书 5.1）。
    """

    recall_block: RecallBlock = Field(default_factory=RecallBlock)
    shadow: Shadow = Field(default_factory=Shadow)
    current_input: str = ""

    @property
    def total_length(self) -> int:
        return self.recall_block.total_length + self.shadow.length + len(self.current_input)

    def assemble(self) -> str:
        """Concatenate recall block, shadow, and current input with section headers.

        将非空的回忆块、残影、当前输入按固定中文标题拼接为多段文本，段间空行分隔，供直接嵌入对话 Prompt。
        """
        parts: list[str] = []
        if self.recall_block.items:
            recall_text = "\n---\n".join(it.content for it in self.recall_block.items)
            parts.append(f"[历史回忆]\n{recall_text}")
        if self.shadow.content:
            parts.append(f"[残影缓冲]\n{self.shadow.content}")
        if self.current_input:
            parts.append(f"[当前输入]\n{self.current_input}")
        return "\n\n".join(parts)
