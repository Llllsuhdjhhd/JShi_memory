from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .interlocutor import InterlocutorAttribution

# 事件与情感量化领域模型；对齐《REMS 记忆系统规范解析》第 1 章（L0/L1~Ln、角色快照、
# 8 维基础情绪、抽象字段、墓碑）及白描遗忘逻辑。


import uuid

def generate_event_id() -> str:
    """Stateless unique event identifier.

    废除字典序，改用无状态分布式的唯一标识（目前使用 uuid4 模拟），
    前缀 ``EVT-``，满足海量并发和高可用等无状态工程需求（白皮书 1.1.1）。
    """
    return f"EVT-{uuid.uuid4().hex}"


class Importance(str, Enum):
    # S = 至关重要（主角、核心行为推动者）
    # A = 非常重要（直接推动事件发展、关键互动）
    # B = 比较重要（有明确互动，但非核心）
    # C = 次要参与（轻度出现或被动关联）
    # D = 可忽略角色（路人、背景板等）
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class EventStatus(str, Enum):
    # 事件在回忆与生命周期中的状态（白皮书 1.1.5）。
    UNCLOSED = "unclosed"
    ACTIVE = "active"
    SILENT = "silent"


# ---------- Emotional model ----------
# 白皮书 2.5：LLM 只输出 8 维基础情绪；arousal / valence 由后端合成。

class BasicEmotionVector(BaseModel):
    """8 basic emotions emitted by the LLM, each normalized to [0, 1]."""

    anger: float = 0.0
    fear: float = 0.0
    joy: float = 0.0
    sadness: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    trust: float = 0.0
    anticipation: float = 0.0

    def clamped(self) -> BasicEmotionVector:
        values = {
            name: min(max(float(getattr(self, name, 0.0)), 0.0), 1.0)
            for name in type(self).model_fields
        }
        return BasicEmotionVector(**values)


class EmotionalModel(BaseModel):
    """Event-local and rolling affective state for one role snapshot."""

    emotion: BasicEmotionVector = Field(default_factory=BasicEmotionVector)
    arousal: float = 0.0
    valence: float = 0.0
    rolling_arousal: float = 0.0
    rolling_valence: float = 0.0
    energy: float = 0.0

    @classmethod
    def from_emotion(cls, emotion: BasicEmotionVector) -> EmotionalModel:
        emotion = emotion.clamped()
        arousal = max(
            emotion.anger,
            emotion.fear,
            emotion.joy,
            emotion.sadness,
            emotion.disgust,
            emotion.surprise,
            emotion.anticipation,
        )
        positive = emotion.joy + emotion.trust
        negative = emotion.anger + emotion.fear + emotion.sadness + emotion.disgust
        valence = (positive - negative) / (positive + negative + 1e-6)
        return cls(
            emotion=emotion,
            arousal=arousal,
            valence=valence,
            rolling_arousal=arousal,
            rolling_valence=valence,
            energy=arousal,
        )


# ---------- Role within an event ----------

class RoleSnapshot(BaseModel):
    # 白皮书 1.1.4：L1 提及、L2 互动白描、L3 决策/意图快照。
    l1_mention: Optional[str] = None
    l2_interaction: Optional[str] = None
    l3_decision: Optional[str] = None


# ---------- Compression budget (白皮书 1.2) ----------

class CompressionBudget(BaseModel):
    """Pre-computed character budgets for each derived-data component.

    在事件封存流程最前端，根据已知的 ``raw_len`` 确定性地计算出各衍生数据项的字符预算，
    并注入后续 LLM 提示词中作为硬约束（白皮书 1.2.2）。
    支持多层级指数级递减预算。
    """
    raw_len: int                   # L0 原文字符长度
    total_budget: int              # raw_len * target_ratio * multiplier
    
    # [L1..L10] 摘要字数上限字典 (key: "L1"..."L10")
    summary_level_budgets: dict[str, int] = Field(default_factory=dict)
    
    # [L1..L3] 角色快照字数上限字典 (key: "L1"..."L3")
    snapshot_level_budgets: dict[str, int] = Field(default_factory=dict)

    wp_budget_per_role: int        # 每个角色白描条目的字数上限
    decoration_budget: int         # 装饰的字数上限
    role_count_estimate: int = 1   # 预估角色数（用于分摍计算）

    @property
    def summary_budget(self) -> int:
        """Backward compatibility: returns L1 budget."""
        return self.summary_level_budgets.get("L1", 0)

    @property
    def snapshot_budget_per_role(self) -> int:
        """Backward compatibility: returns L3 (base) budget."""
        return self.snapshot_level_budgets.get("L3", 0)


class EventRoleEntry(BaseModel):
    # 单个角色在一次事件中的瞬时记录（之后会被写入角色白描时间线）。
    role_id: str
    is_subject: bool = False  # True = 匠石（主体）；情感只属于主体（design/410）
    importance: Importance = Importance.C
    role_snapshot: RoleSnapshot = Field(default_factory=RoleSnapshot)
    emotional_model: EmotionalModel = Field(default_factory=EmotionalModel)


# ---------- Event ----------
# 基本事件与抽象事件共用同一结构；抽象事件 is_abstract=True 且带 abstraction_level/source_events。

class Event(BaseModel):
    event_id: str = Field(default_factory=generate_event_id)
    subject_id: str = ""  # 记忆单元归属主体（匠石）；由 ingest_batch 注入（design/810）
    create_time: datetime = Field(default_factory=datetime.now)
    input_id: Optional[str] = None  # 记录产生此事件的经历段 segment_id（30 侧；design/210）
    source_ids: list[str] = Field(default_factory=list)  # 来源链（design/210）
    occurred_at: Optional[datetime] = None  # 经历发生时间；空 = 摄入时刻（design/210/810）
    content_raw: str  # L0：基本事件为已闭环事实原文；抽象事件为自子事件归纳的合成描述（白皮书 1.1.2、3.1）。

    summaries: dict[str, str] = Field(default_factory=dict)  # L1~Ln 递归摘要（白皮书 1.1.3）。
    summary_lengths: dict[str, int] = Field(default_factory=dict)
    actual_max_level: int = 0  # 熔断后的实际最高摘要层级。

    role_list: list[EventRoleEntry] = Field(default_factory=list)  # 当次事件涉及角色与情感快照（1.1.4）。
    interlocutor_attributions: list[InterlocutorAttribution] = Field(
        default_factory=list,
        description="历史对话对象及来源经历段；一个封存事件可包含多个来源段/对话对象",
    )

    is_abstract: bool = False  # 架构级区分基本/抽象事件（1.1.5、3.1）。
    is_abstracted: bool = False  # 是否已被更高阶认知吸收（1.1.5）。
    # 被更高层抽象「覆盖吸收」的累计强度（非负）；用于回忆 RRF 分上乘以 exp(-decay*coverage)。
    abstract_coverage: float = 0.0
    status: EventStatus = EventStatus.ACTIVE # unclosed/active/silent（1.1.5）。

    decoration: Optional[str] = None  # 非事实装饰，支撑「梦见」等（1.1.6）。
    insight: Optional[str] = None  # 基本事件宜为空；抽象事件存规律；墓碑审计可追加（1.1.6、4.3）。
    event_length: int = 0  # L0 字符长度，用于约束与检索预估（1.1.7）。

    # 仅抽象事件：演化树深度与证据链 source_events（白皮书 1.1.8、3.1）。
    abstraction_level: Optional[int] = None
    source_events: Optional[list[str]] = None

    # 墓碑化：逻辑上被修正覆盖，保留审计但回忆排除（白皮书 4.3）。
    is_tombstoned: bool = False

    # 激活能量：新版以角色 arousal / rolling energy 聚合得到，用于检索与审计。
    activation_energy: float = 0.0

    # 动态压缩率（白皮书 1.2）：封存后实际的 sum_len / raw_len，用于审计与追踪。
    compression_ratio: float = 0.0

    # ---- 80/20 强制分裂链路（2026-05 新增） ----
    # 若本事件是某次强制分裂的"前缀"（边界模型把过长未完成切成 prefix_completed + tail_uc），
    # 则 ``split_successor_event_ids`` 在其对应 tail UC 闭环为事件时追加该事件 id。
    # 允许 len>1：一条超长叙事可能跨多轮被切多次，前缀的后继可能再被切。
    split_successor_event_ids: list[str] = Field(default_factory=list)
    # 若本事件是 tail UC 闭环而来，这里继承 UC 的 ``split_prefix_event_ids`` 链。
    # 回忆时 RecallService 会在命中本事件后自动把这些前缀事件拉进回忆块（允许降档压缩，但不丢弃）。
    split_prefix_event_ids: list[str] = Field(default_factory=list)

    # §4.7.3 情感极值免死金牌：禁止摘要降级与 ASF 遮蔽。
    ptsd_immune: bool = False
    # 事件来源：external | internal | dream（字符串，保留扩展；legacy normal 视为 external）。
    origin: str = "external"

    # 事件级主体情感（匠石在事件中的情感；对象无情感，见 design/410）。
    emotion: Optional[EmotionalModel] = None

    # 记忆单元级遗忘因子（情感 arousal 初始化，随时间衰减；回忆命中强化，见 design/610/1010）。
    forgetting_factor: float = 1.0

    # 封存原因：closed 自然闭环 / split 资源切开的前段 / truncated 搁置或无法切开。
    seal_reason: str = "closed"

    # Optional recall metadata (BM25 / literary profile); persisted as recall_metadata JSON.
    keywords: list[str] = Field(default_factory=list)
    location: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.event_length:
            self.event_length = len(self.content_raw)

    # ------------------------------------------------------------------
    # Affective helpers
    # ------------------------------------------------------------------

    @property
    def affective_energy(self) -> float:
        """Event-level affective intensity, defined as max role arousal."""
        if not self.role_list:
            return 0.0
        return min(max(entry.emotional_model.arousal for entry in self.role_list), 1.0)

    @property
    def event_valence(self) -> float:
        """Mean event valence across role snapshots."""
        if not self.role_list:
            return 0.0
        total = sum(entry.emotional_model.valence for entry in self.role_list)
        return total / len(self.role_list)

    @property
    def rolling_valence(self) -> float:
        """Mean rolling valence across role snapshots."""
        if not self.role_list:
            return 0.0
        total = sum(entry.emotional_model.rolling_valence for entry in self.role_list)
        return total / len(self.role_list)

    @property
    def mid_summary_key(self) -> str:
        """Return the key for the middle-level summary (used by recall / abstraction).

        将 ``summaries`` 的键按 L1、L2… 数值排序后取中位下标对应的级别，作为「中间粒度」摘要键；
        若无任何摘要则与实现一致地退化为 ``L1``。供回忆组装时在长度预算下做懒索引降级，
        以及归纳技能非锚定路径选用。
        """
        if not self.summaries:
            return "L1"
        levels = sorted(self.summaries.keys(), key=lambda k: int(k[1:]))
        return levels[len(levels) // 2]
