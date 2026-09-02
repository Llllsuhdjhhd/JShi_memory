"""人物肖像模型：per-object 10 级渐进人物摘要 + 记忆疲态可见性。

规格（design/1010 §8.4 之后的 portrait 子项）：
- 10 级，从最简(L1)到最详细(Ln)，每级是上一级 growth(默认2) 倍；
- L1 = min(portrait_magic_num, 总长 / growth^(max_levels-1))；
- 某级预算 > 总长 即停，保留该级原长度；最多 max_levels 级；
- 压缩率 = 汇总原始字数 / 最长一级字数；
- 记忆疲态 fatigue ∈ (0,1]，默认 1（无遗忘）；<1 时按概率让部分摘要/事件概率不可见（期望可见占比 ≈ fatigue），
  且被隐藏部分仍带偶然性（低权重也有非零可见概率）。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PortraitLevelInfo(BaseModel):
    """一级人物摘要：级别名 + 文本 + 预算/实际字数。"""

    level: str            # "L1".."L10"
    text: str
    budget: int           # 该级预算字数
    actual_len: int = 0   # 生成后实际字数


class ObjectPortrait(BaseModel):
    """per-object 人物肖像：从最简到最详细的多级渐进人物摘要。"""

    subject_id: str = ""
    object_id: str
    name: str | None = None
    levels: dict[str, PortraitLevelInfo] = Field(default_factory=dict)  # {"L1": ...}
    max_level: str = ""            # 当前最长级（Lk 中 k 最大者）
    total_content_len: int = 0     # 累积总长（增量累积，不每次重算）
    compression_ratio: float = 0.0  # 汇总原始字数 / 最长一级字数
    fatigue: float = 1.0           # 记忆疲态（可见性期望占比）
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def longest_level(self) -> PortraitLevelInfo | None:
        """当前最长（最详细）的一级，作为"已有内容"用于增量压缩。"""
        if not self.max_level:
            return None
        return self.levels.get(self.max_level)

    def visible_summary(self, fatigue: float | None = None) -> str:
        """按疲态取可见部分：以最长级的文本作为完整人设（召回/展示用）。"""
        ll = self.longest_level
        text = ll.text if ll else ""
        # 疲态 <1 时不是硬切，而是概率可见；此处返回文本本身，可见性由调用方按 fatigue 抽样决定。
        return text


class PendingPortraitSummary(BaseModel):
    """一条对象人物摘要（待并入画像或已并入）。

    一次记忆落位涉及多个对象时，**每个对象都会生成一条自己的摘要**（主体对该对象的记忆），
    汇入该对象的摘要集合。``incorporated`` 标记是否已并入肖像（全量重建用全部，增量用最长级+未并入）。
    """

    summary_id: str
    object_id: str
    subject_id: str = ""
    text: str
    source_event_id: str | None = None
    weight: float = 0.5           # 重要性/近因合成的可见性权重 [0,1]
    incorporated: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryFatigue:
    """记忆疲态：可见性期望占比 (0,1]，默认 1（无遗忘）。

    **有两种疲态，不能共用一个**：
    - 事件：仅由整体记忆负担决定（``event_fatigue``），不看对象负担；
    - 人物肖像摘要：由整体负担 + 对象负担合成（``portrait_fatigue``）。
    本类是一个可见性计算器，两个疲态各用一个实例；``fatigue`` 是当前值。

    ``visible_probability``：``p = fatigue * (floor + (1-floor) * weight)``，weight ∈ [0,1]。
    - 高权重(近因/重要) → p 接近 fatigue；低权重 → p = floor*fatigue（>0，保留偶然性）；
    - 期望可见占比 ≈ fatigue；没有任何一条是 0% 可见（低权重也偶发可见）。
    """

    def __init__(self, fatigue: float = 1.0, overall_burden: float = 1.0, object_burden: float = 1.0):
        self.fatigue = max(0.0, min(1.0, fatigue))
        self.overall_burden = max(0.0, min(1.0, overall_burden))
        self.object_burden = max(0.0, min(1.0, object_burden))

    @classmethod
    def from_overall(cls, overall_burden: float) -> "MemoryFatigue":
        """事件疲态：仅看整体负担。burden 越大 → fatigue 越小（遗忘越多）。"""
        burden = max(0.0, min(1.0, overall_burden))
        fatigue = 1.0 - burden
        return cls(fatigue=max(0.01, fatigue), overall_burden=burden, object_burden=0.0)

    @classmethod
    def from_combined(cls, overall_burden: float, object_burden: float) -> "MemoryFatigue":
        """肖像疲态：整体 + 对象负担合成（几何平均，可换加权/调和）。"""
        o = max(0.0, min(1.0, overall_burden))
        ob = max(0.0, min(1.0, object_burden))
        burden = (o * ob) ** 0.5 if o > 0 and ob > 0 else 1.0
        fatigue = 1.0 - max(0.0, min(1.0, burden))
        return cls(fatigue=max(0.01, fatigue), overall_burden=o, object_burden=ob)

    @classmethod
    def manual(cls, fatigue: float) -> "MemoryFatigue":
        """暂用人工设置（配置直接给值）。"""
        return cls(fatigue=max(0.0, min(1.0, fatigue)))

    @staticmethod
    def visible_probability(weight: float, fatigue: float, floor: float = 0.2) -> float:
        weight = max(0.0, min(1.0, weight))
        fatigue = max(0.0, min(1.0, fatigue))
        return max(0.0, min(1.0, fatigue * (floor + (1.0 - floor) * weight)))

    def is_visible(self, weight: float, *, rng=None, floor: float = 0.2) -> bool:
        """按可见概率抽样一次（偶然性：低权重也可能可见）。"""
        import random as _r
        rng = rng or _r
        return rng.random() < self.visible_probability(weight, self.fatigue, floor)

    @staticmethod
    def apply_fatigue(weights: list[float], fatigue: float, *, rng=None) -> list[bool]:
        """集合级可见性：按权重排序 + 软分界，使**整体可见占比 ≈ fatigue**（期望值）。

        - 排在前 ``fatigue`` 比例的条目（高权重）大概率可见；
        - 其余（低权重）小概率可见（保留偶然性，不是硬切）；
        - 权重越高越易可见。返回与 ``weights`` 同序的 bool 列表。
        """
        import random as _r
        rng = rng or _r
        n = len(weights)
        if n == 0:
            return []
        fatigue = max(0.0, min(1.0, fatigue))
        if fatigue >= 1.0:
            return [True] * n            # 无遗忘：全部可见
        order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        k = int(round(fatigue * n))
        visible = [False] * n
        for rank, idx in enumerate(order):
            if rank < k:
                p = 0.9 + 0.1 * (1.0 - (rank / max(1, n - 1))) if n > 1 else 0.95
            else:
                p = 0.1 * (1.0 - ((rank - k) / max(1, n - k))) if (n - k) > 1 else 0.05
            visible[idx] = rng.random() < p
        return visible


__all__ = [
    "MemoryFatigue",
    "ObjectPortrait",
    "PendingPortraitSummary",
    "PortraitLevelInfo",
]
