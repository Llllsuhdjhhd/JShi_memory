from __future__ import annotations

import logging

# 递归摘要 L1→Ln：保真压缩与逐级抽象，直至某级字数 ≤ summary_fuse_min_chars 时熔断（白皮书 1.1.3）。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import SUMMARY_SYSTEM, SUMMARY_USER

logger = logging.getLogger(__name__)


class SummaryResult(BaseModel):
    summaries: dict[str, str] = Field(default_factory=dict)
    summary_lengths: dict[str, int] = Field(default_factory=dict)
    actual_max_level: int = 0


class SummaryGenerationSkill:
    """Recursively generates L1…Ln summaries and populates ``Event`` summary fields.

    对 ``content_raw`` 迭代调用 LLM：每一级基于上一级输出继续压缩，直到单级字符数低于
    ``summary_fuse_min_chars``（含等号：该级及更短则停）或达到 ``max_levels``。结果写入 ``summaries``、``summary_lengths``、
    ``actual_max_level``（白皮书 1.1.3 与动态熔断）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def generate(self, content_raw: str, *, max_levels: int = 10, budget: "CompressionBudget" | None = None) -> SummaryResult:
        """Recursively generate L1 … Ln summaries from ``content_raw``.

        Stops when the new summary's character count is at most ``summary_fuse_min_chars``
        or ``max_levels`` is reached.

        从 ``content_raw`` 递归生成 L1…Ln：当新摘要长度 **≤** ``summary_fuse_min_chars`` 或达到 ``max_levels`` 时停止；
        返回各级文本与字数及实际层数。使用 ``budget`` 中的指数衰减预算表（白皮书 1.2）。
        默认 ``max_levels=10`` 对齐白皮书 §1.1.3 的 L1–L10 体系；熔断触发时实际层数会少于 10。
        """
        summaries: dict[str, str] = {}
        lengths: dict[str, int] = {}
        current_text = content_raw
        source_label = "L0 原文"

        for level in range(1, max_levels + 1):
            target = f"L{level}"

            # 注入当前级别的指数级衰减字数预算
            level_budget = 0
            if budget and target in budget.summary_level_budgets:
                level_budget = budget.summary_level_budgets[target]
            
            if level_budget > 0:
                budget_hint = f"【字数预算】请将本级摘要严格控制在 {level_budget} 字以内。\n"
            else:
                budget_hint = ""

            sys_msg = SUMMARY_SYSTEM.format(fuse_min_chars=self._config.summary_fuse_min_chars)
            user_msg = SUMMARY_USER.format(
                source_level=source_label,
                target_level=target,
                text=current_text,
                budget_hint=budget_hint,
            )

            data = self._llm.complete_json(
                "summary",
                [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
            )

            summary_text = data.get("summary", "")
            char_count = data.get("char_count", len(summary_text))

            if not summary_text:
                break

            summaries[target] = summary_text
            lengths[target] = char_count

            if char_count <= self._config.summary_fuse_min_chars:
                logger.debug("Summary fuse triggered at %s (%d chars)", target, char_count)
                break

            current_text = summary_text
            source_label = target

        return SummaryResult(
            summaries=summaries,
            summary_lengths=lengths,
            actual_max_level=len(summaries),
        )
