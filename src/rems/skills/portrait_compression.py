"""人物肖像压缩 Skill（design/1010 §8.4 portrait）。

- ``generate_object_summary``：一次记忆落位涉及多个对象时，为**每个对象**生成一条人物摘要
  （主体对该对象的记忆）。
- ``compress_incremental``：用「已有肖像的最长级 + 新摘要集」压缩成新多级（增量式）。
- ``compress_full``：用「该对象所有摘要」重建多级（全量式）。

模型驱动的压缩（可忽略/简化/调整）；无 LLM 时退化为确定性拼接（保底）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import REMSConfig
    from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class PortraitCompressionSkill:
    def __init__(self, llm: "LLMProvider | None", config: "REMSConfig"):
        self._llm = llm
        self._config = config

    # ------------------------------------------------------------------
    def generate_object_summary(self, event, object_id: str, name: str | None = None) -> str:
        """为 (event, object) 生成一条人物摘要：主体记忆里关于该对象的一段话。

        多对象事件：每个对象各生成一条，视角是"主体记得的该对象"。
        """
        label = name or object_id
        if self._llm is None:
            return self._fallback_summary(event, object_id, label)
        try:
            prompt = (
                "你正在为「匠石」整理对一个人物/对象的记忆。基于下面这段经历，只写"
                f"「匠石对『{label}』（object_id={object_id}）的印象/记忆」，一句话，"
                "聚焦该对象本身，不要混入其他对象。\n\n经历：\n"
                f"{event.content_raw if hasattr(event, 'content_raw') else event}\n"
                "返回纯文本，20-80字。"
            )
            out = self._llm.complete("portrait_summary", [{"role": "user", "content": prompt}])
            return (out or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("portrait object summary failed: %s", exc)
            return self._fallback_summary(event, object_id, label)

    # ------------------------------------------------------------------
    def _fallback_summary(self, event, object_id: str, label: str) -> str:
        """无 LLM 兜底：截取事件原文里提及该对象附近的一小段（确定性）。

        找不到对象 id 时退回事件原文截断，保证"人物摘要"有实质内容。
        """
        raw = event.content_raw if hasattr(event, "content_raw") else str(event)
        idx = raw.find(object_id)
        if idx < 0:
            idx = raw.find(label)
        if idx >= 0:
            start = max(0, idx - 30)
            return raw[start: idx + 70].strip() or raw[:60]
        return (raw[:60].strip() or f"（记忆涉及 {label}）")

    # ------------------------------------------------------------------
    def compress(
        self,
        budgets: dict[str, int],
        *,
        existing_longest: str = "",
        new_summaries: list[str],
        all_summaries: list[str],
        mode: str = "incremental",
    ) -> dict[str, str]:
        """按预算把源材料压缩成多级文本。mode = incremental | full。"""
        source = new_summaries
        context = existing_longest
        if mode == "full":
            source = all_summaries
            context = ""
        if self._llm is None:
            return self._fallback_compress(budgets, context, source)
        try:
            prompt = self._compress_prompt(budgets, context, source, mode)
            out = self._llm.complete("portrait_compress", [{"role": "user", "content": prompt}])
            return self._parse_levels(out, budgets)
        except Exception as exc:  # noqa: BLE001
            logger.warning("portrait compress failed: %s", exc)
            return self._fallback_compress(budgets, context, source)

    def _compress_prompt(self, budgets, context, source, mode) -> str:
        base = "你有「匠石」对某人/某物的一系列记忆片段，请按层级压缩成人物肖像。\n"
        if mode == "incremental":
            base += f"已有肖像（最长级）：\n{context}\n---\n新增摘要：\n{chr(10).join(source)}\n"
        else:
            base += f"全部摘要：\n{chr(10).join(source)}\n"
        base += "请按下面各级字数预算，逐级生成（每级是上一级的压缩；L1 最简、Ln 最详细），"
        "忽略/简化/调整，只输出若干行，每行形如 `L1: <文本>`：\n"
        base += "\n".join(f"{k}: {v} 字以内" for k, v in budgets.items())
        return base

    def _parse_levels(self, out: str, budgets: dict[str, int]) -> dict[str, str]:
        levels: dict[str, str] = {}
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            for key in budgets:
                if line.startswith(f"{key}:"):
                    text = line[len(f"{key}:"):].strip()
                    levels[key] = text[: budgets[key]]
                    break
        return levels

    def _fallback_compress(self, budgets, context, source) -> dict[str, str]:
        """确定性兜底：把源材料拼接后按预算截断分配（无 LLM 时）。"""
        parts = []
        if context:
            parts.append(context)
        parts.extend(source)
        material = "\n".join(parts)
        levels = {}
        for key, budget in budgets.items():
            levels[key] = material[:budget] if material else ""
        return levels


__all__ = ["PortraitCompressionSkill"]
