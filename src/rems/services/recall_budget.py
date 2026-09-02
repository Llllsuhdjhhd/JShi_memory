"""回忆预算管理（design/1010 §8.5）。

预算 = 本次回忆的总字数，与回忆等级相关。
- 回忆分 `recall_level_count`（默认 6）级，各级字数递增：L1=最短=recall_magic_num，之后逐级 ×growth，L{count}=最长。
- 分配：条目按评分排序，贪心——能装最高级就装最高级，装不下下降级，再装不下丢弃。
- 对话人的人物肖像作为回忆条目同场参与；总字数 ≤ recall_budget_chars（0 => physical_redline）。
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import REMSConfig

logger = logging.getLogger(__name__)


def recall_level_budgets(magic_num: int = 100, level_count: int = 6, growth: float = 2.0) -> dict[str, int]:
    """6 级回忆等级预算：L1=最短(magic)，逐级 ×growth，L{count}=最长。"""
    return {
        f"L{k}": max(1, int(magic_num * (growth ** (k - 1))))
        for k in range(1, level_count + 1)
    }


class RecallBudgetManager:
    """把已排序的回忆条目分配到 6 级之一，使拼装总字数 ≤ 预算。"""

    def __init__(self, config: REMSConfig):
        self._config = config

    # ------------------------------------------------------------------
    def level_budgets(self) -> dict[str, int]:
        return recall_level_budgets(
            self._config.recall_magic_num,
            self._config.recall_level_count,
            self._config.recall_level_growth,
        )

    def budget_chars(self) -> int:
        """本次回忆总预算：显式 recall_budget_chars，否则用 physical_redline。"""
        if self._config.recall_budget_chars:
            return int(self._config.recall_budget_chars)
        return self._config.physical_redline

    # ------------------------------------------------------------------
    def allocate(
        self,
        items: list[Any],
        *,
        budget: int | None = None,
        sort_key=None,
    ) -> list[dict[str, Any]]:
        """贪心分配：按评分降序，给每条分配"能装下的最高级"，装不下下降级，再装不下丢弃。

        ``items`` 为本回忆的候选（事件/人物肖像），每条需有 ``score``（评分/重要性）与内容文本。
        返回 ``[{fragment, level, budget, used}]``，其中 ``budget`` 为该级字数上限，``used``=实际截断长度。
        """
        budget = budget or self.budget_chars()
        budgets = self.level_budgets()
        # 从最长级(L{count}) 到最短级(L1) 排序
        desc = sorted(budgets.items(), key=lambda kv: kv[1], reverse=True)
        ranked = sorted(items, key=sort_key or (lambda it: getattr(it, "score", 0.0)), reverse=True)

        remaining = budget
        selected: list[dict[str, Any]] = []
        for item in ranked:
            text = getattr(item, "content", None) or getattr(item, "text", None) or ""
            clen = len(text)
            chosen = None
            for level, cap in desc:
                use = min(cap, clen)          # 实际展示字数 = 档位上限 与 内容长度 的较小值
                if use <= remaining:
                    chosen = (level, cap, use)
                    break
            if chosen is None:
                break  # 连最短级(甚至是内容本身)都装不下 -> 丢弃后续条目
            remaining -= chosen[2]
            selected.append({
                "fragment": item,
                "level": chosen[0],
                "budget": chosen[1],
                "used": chosen[2],
            })
        return selected


__all__ = ["RecallBudgetManager", "recall_level_budgets"]
