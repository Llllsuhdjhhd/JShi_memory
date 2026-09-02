"""人物肖像服务（design/1010 §8.4 portrait）。

负责 per-object 10 级渐进人物摘要：
- 等级化：``compute_level_budgets``（L1=max(魔法数100, 总长/2^9)，每级 growth 倍，超总长即停、截到总长、最多 10 级）；
- 触发：某对象「待并入」人物摘要累计字数 >= context_window / portrait_trigger_divisor 时执行肖像更新；
- 压缩：增量式（已有肖像最长级 + 新摘要）或全量式（所有摘要）两种方式，由模型/兜底 Skill 完成；
- 疲态：事件用 ``event_fatigue``（仅整体负担）、肖像用 ``portrait_fatigue``（整体+对象），按概率可见（保留偶然性）。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from ..config import REMSConfig
from ..models.portrait import (
    MemoryFatigue,
    ObjectPortrait,
    PendingPortraitSummary,
    PortraitLevelInfo,
)
from ..skills.portrait_compression import PortraitCompressionSkill
from ..storage.repository import PortraitRepository

logger = logging.getLogger(__name__)


def compute_level_budgets(
    total_len: int,
    *,
    magic_num: int = 100,
    max_levels: int = 10,
    growth: float = 2.0,
) -> dict[str, int]:
    """按规格计算多级预算。

    ``L1 = max(magic_num, 总长 / growth^(max_levels-1))`` —— 取较长，保最短级有效（≥魔法数）；
    ``Lk = L1 * growth^(k-1)``；某级预算 > 总长 即停（该级截到总长），最多 max_levels 级。
    """
    if total_len <= 0:
        return {}
    base = max(int(magic_num), int(total_len / (growth ** (max_levels - 1))))
    base = max(1, base)
    budgets: dict[str, int] = {}
    for k in range(1, max_levels + 1):
        b = int(base * (growth ** (k - 1)))
        if b > total_len:
            break  # 某一级预算 > 总长 → 暂停，不再往上加；已保留各级原长度。
        budgets[f"L{k}"] = b
    return budgets


class PortraitService:
    def __init__(
        self,
        config: REMSConfig,
        repo: PortraitRepository,
        skill: PortraitCompressionSkill | None = None,
    ):
        self._config = config
        self._repo = repo
        self._skill = skill or PortraitCompressionSkill(None, config)

    # ------------------------------------------------------------------
    # 等级化（确定性，可直接单测）
    # ------------------------------------------------------------------
    def level_budgets(self, total_len: int) -> dict[str, int]:
        return compute_level_budgets(
            total_len,
            magic_num=self._config.portrait_magic_num,
            max_levels=self._config.portrait_max_levels,
            growth=self._config.portrait_growth,
        )

    # ------------------------------------------------------------------
    # 疲态（两套：事件=仅整体负担；肖像=整体+对象）
    # ------------------------------------------------------------------
    def event_fatigue(self) -> MemoryFatigue:
        """事件视角疲态：仅由整体记忆负担决定（此处暂从配置 event_fatigue 取）。"""
        return MemoryFatigue.manual(self._config.event_fatigue)

    def portrait_fatigue(self, object_burden: float | None = None) -> MemoryFatigue:
        """肖像视角疲态：整体 + 对象负担合成；暂从配置 portrait_fatigue 取。"""
        return MemoryFatigue.manual(self._config.portrait_fatigue)

    # ------------------------------------------------------------------
    # note_event：一次记忆落位，对每个对象各生成一条人物摘要并入缓冲
    # ------------------------------------------------------------------
    def note_event(self, event, object_ids: list[str]) -> dict[str, bool]:
        """为 event 涉及的每个对象生成人物摘要。

        返回 {object_id: generated}。对象缓中累计字数超过触发阈值即执行该对象肖像更新。
        """
        triggered: dict[str, bool] = {}
        for oid in object_ids:
            if not oid:
                continue
            name = self._obj_name(oid)
            text = self._skill.generate_object_summary(event, oid, name)
            if not (text or "").strip():
                continue
            item = PendingPortraitSummary(
                summary_id=uuid.uuid4().hex,
                object_id=oid,
                subject_id=getattr(event, "subject_id", "") or "",
                text=text.strip(),
                source_event_id=getattr(event, "event_id", None),
                weight=self._weight_for(event, oid),
            )
            self._repo.add_summary(item)
            triggered[oid] = self._maybe_update(oid)
        return triggered

    # ------------------------------------------------------------------
    # 肖像更新（两种压缩方式）
    # ------------------------------------------------------------------
    def update_portrait(self, object_id: str) -> bool:
        prev = self._repo.get_portrait(object_id)
        pending = self._repo.list_summaries(object_id, incorporated=False)
        all_summaries = self._repo.list_summaries(object_id, incorporated=True) + pending
        if not pending and prev is None:
            return False

        # 总长（增量累积）：已有 total + 待并入
        total_len = (prev.total_content_len if prev else 0) + sum(len(s.text) for s in pending)
        budgets = self.level_budgets(total_len)
        if not budgets:
            return False

        mode = self._config.portrait_compose_mode
        if mode == "auto":
            # 策略2(所有摘要) 仅当累计总长在「1/6 上下文」以内；否则用策略1(增量)。
            threshold = int(self._config.context_window * self._config.portrait_full_mode_context_ratio)
            mode = "full" if total_len <= threshold else "incremental"
        existing_longest = prev.longest_level.text if prev and prev.longest_level else ""
        levels_text = self._skill.compress(
            budgets,
            existing_longest=existing_longest,
            new_summaries=[s.text for s in pending],
            all_summaries=[s.text for s in all_summaries],
            mode=mode,
        )
        if not levels_text:
            return False

        levels: dict[str, PortraitLevelInfo] = {}
        for k, text in levels_text.items():
            budget = budgets.get(k, 0)
            levels[k] = PortraitLevelInfo(level=k, text=text, budget=budget, actual_len=len(text))

        max_key = max(levels.keys(), key=lambda k: int(k[1:])) if levels else ""
        longest = levels.get(max_key)
        ratio = 0.0
        if longest and longest.actual_len:
            ratio = round(total_len / longest.actual_len, 4)

        portrait = ObjectPortrait(
            subject_id=(prev.subject_id if prev else self._subject_of(all_summaries)),
            object_id=object_id,
            name=self._obj_name(object_id),
            levels=levels,
            max_level=max_key,
            total_content_len=total_len,
            compression_ratio=ratio,
            fatigue=self._config.portrait_fatigue,
            created_at=prev.created_at if prev else datetime.now(),
            updated_at=datetime.now(),
        )
        self._repo.save_portrait(portrait)
        self._repo.mark_incorporated([s.summary_id for s in pending])
        logger.info("portrait updated for %s: levels=%s total=%d ratio=%.3f",
                    object_id, sorted(levels.keys()), total_len, ratio)
        return True

    def get_portrait(self, object_id: str) -> ObjectPortrait | None:
        return self._repo.get_portrait(object_id)

    # ------------------------------------------------------------------
    def _maybe_update(self, object_id: str) -> bool:
        pending = self._repo.list_summaries(object_id, incorporated=False)
        trigger = int(self._config.context_window / self._config.portrait_trigger_divisor)
        total_pending = sum(len(s.text) for s in pending)
        if total_pending >= trigger:
            return self.update_portrait(object_id)
        return False

    def _obj_name(self, object_id: str) -> str | None:
        # 名字由 01 侧映射；此处可尝试从 object_memory_entries 读取，暂返回 None。
        return None

    def _weight_for(self, event, object_id: str) -> float:
        # 重要性/近因权重 → 可见性权重 [0,1]；首版按事件激活能量近似。
        ae = getattr(event, "activation_energy", 0.0) or 0.0
        return min(max(ae, 0.0), 1.0)

    def _subject_of(self, summaries: list[PendingPortraitSummary]) -> str:
        for s in summaries:
            if s.subject_id:
                return s.subject_id
        return ""


__all__ = ["PortraitService", "compute_level_budgets"]
