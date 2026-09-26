from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# REMS 全局配置（对照《REMS 记忆系统规范解析》白皮书）。
# - len_msg / physical_redline / safe_watermark：1.1.7 事件长度与物理防御、4.2 触发与截断；
# - abstract_subset_*, abstract_coverage_*：3.2 频繁子集与覆盖度检索降权；
# - ae_*、wp_*：1.1.4、2.2 情感能量（AE）与白描动态遗忘；
# - hallucination_anchor_prob：兼容旧配置；当前抽象合成始终使用叶子基本事件 content_raw；
# - tombstone_prefix：4.3 墓碑化时在 insight 中的审计标记前缀；
# - user_mode / core_user_role_id / active_participants：2.2 单人/多人模式与提示词注入引擎。
# 加载：环境变量前缀 REMS_，嵌套键用 __ 分隔；可选 .env。


class UserMode(str, Enum):
    """Global pronoun-resolution mode (白皮书 2.2).

    SINGLE：单人模式——第一人称与无主语动作默认归属 ``core_user_role_id``，降低代词消解幻觉。
    MULTI：多人模式——提供 ``active_participants`` 花名册，模型需做多方共指消解，禁止默认归一。
    """

    SINGLE = "single"
    MULTI = "multi"


class TaskModelMapping(BaseModel):
    """Maps each REMS skill/task type to a specific LLM model name.

    将 REMS 中各技能任务类型（summary、boundary_detection、role_extraction、abstraction、insight 等）
    映射到具体模型名称字符串，便于在成本、延迟与能力之间做分流；未识别类型回退 ``default``。
    """

    # 以下为各技能默认模型名；可按任务强度分流成本（摘要/边界/抽取/抽象等）。

    summary: str = "deepseek-v4-flash"
    boundary_detection: str = "deepseek-v4-flash"
    # 超长未完成事件的分裂修复；通常可直接复用 boundary_detection 的模型，也可指定更强模型。
    overlong_uc_split: str = "deepseek-v4-flash"
    # 可选：对 boundary_detection 输出的二级 LLM 评估器；默认同主模型。
    boundary_evaluation: str = "deepseek-v4-flash"
    role_extraction: str = "deepseek-v4-flash"
    # 事件充实：一次调用产出摘要 + 角色；与 summary 同主模型时便于在 DashScope 侧统一配额。
    event_enrichment: str = "deepseek-v4-flash"
    abstraction: str = "deepseek-v4-flash"
    # 抽象前叙事同一性自检；可与 abstraction 共用模型。
    narrative_coherence: str = "deepseek-v4-flash"
    # 回忆块与当前输入的相关性抽检（低频）。
    recall_block_relevance: str = "deepseek-v4-flash"
    # 向量检索前将长 query 压缩为 act 检索摘要。
    recall_query_compress: str = "deepseek-v4-flash"
    insight: str = "deepseek-v4-flash"
    default: str = "deepseek-v4-flash"


class LLMConfig(BaseModel):
    # 兼容 OpenAI 协议的对话 API（DeepSeek / DashScope 等）。密钥与基址用环境变量覆盖（见 .env.example）。
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    task_models: TaskModelMapping = Field(default_factory=TaskModelMapping)
    temperature: float = 0.3
    max_retries: int = 3


class EmbeddingConfig(BaseModel):
    # pydantic v2 的 ``model_`` 命名空间会对 ``model_name`` 产生 UserWarning，这里显式放通。
    model_config = {"protected_namespaces": ()}

    # 向量嵌入：默认本地 SentenceTransformer；可扩展为远程 API。
    provider: str = "local"
    model_name: str = "BAAI/bge-small-zh-v1.5"
    api_base_url: str | None = None
    api_key: str | None = None


class StorageConfig(BaseModel):
    # 关系型数据（事件/角色/代谢状态）与 Qdrant 向量库。
    database_url: str = "sqlite:///rems.db"
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "rems_events"
    qdrant_path: str | None = None


class Tier1Config(BaseModel):
    active_pool_base: int = 100_000
    active_pool_min: int = 20_000
    sample_key_refresh_interval_s: int = 3600


class TriBandConfig(BaseModel):
    vector_dim_act: int = 512
    vector_dim_emo: int = 512
    vector_dim_ent: int = 512
    weight_fact: tuple[float, float, float] = (0.7, 0.1, 0.2)
    weight_emotion: tuple[float, float, float] = (0.2, 0.6, 0.2)
    weight_entity: tuple[float, float, float] = (0.1, 0.0, 0.9)
    # Act-band ingest/query alignment (§4.4.0)
    act_short_raw_chars: int = 800
    act_short_level: str = "L1"
    act_long_level: str = "L2"
    act_max_chars: int = 4000
    # bge-small-zh 512 token 安全上限（入库 act embed）
    act_embed_max_chars: int = 480
    # 查询 act：LLM 压缩目标字数与硬上限（fallback 截断亦用 target）
    query_act_max_chars: int = 200
    query_act_compress_max_chars: int = 400
    query_act_use_tail: bool = True
    ent_names_only: bool = True
    emo_search_enabled: bool = False


class LifecycleConfig(BaseModel):
    bypass_confidence_threshold: float = 0.5
    bypass_rate_limit_per_minute: int = 1
    ptsd_arousal_threshold: float = 0.8
    dream_enabled: bool = False
    dream_batch_size: int = 5
    shadow_compaction_fragment_threshold: int = 8


class REMSConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REMS_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # 由 token 窗口折算字符预算（白皮书 1.1.7：单条与缓冲区与上下文比例关系）。
    # 默认 88000：以 chars_per_token=1.5 折算，使 len_msg = 88000 * 1.5 / 66 ≈ 2000。
    context_window: int = 88000
    chars_per_token: float = 1.5

    # 记忆主体（匠石）：恒为单个主体值；30 侧在 MemoryBatch 提供（design/210）。
    subject_id: str = ""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    tier1: Tier1Config = Field(default_factory=Tier1Config)
    tri_band: TriBandConfig = Field(default_factory=TriBandConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)

    # ---- Recall: role-aware summary tier selection (白皮书 4.4 Lazy Index) ----
    # 默认召回摘要起点：0 = 中间级（mid），-N = 向 L1 偏移 N 档（更精细），+N = 向最高级偏移 N 档（更压缩）。
    recall_default_tier_offset: int = 0
    # 关注角色（S/A 重要性）：在默认档基础上向 L1 方向再偏移的档数（更精细）。
    recall_primary_role_detail_shift: int = 1
    # 次要/无关角色（C/D 重要性或未出现）：在默认档基础上向高压缩方向偏移的档数。
    recall_minor_role_compress_shift: int = 1
    # 摘要最低字数门槛：某档摘要字数 ≤ 此值时视为已最大压缩，不再向上推进档位。
    recall_summary_min_chars: int = 20
    
    # ---- Recall Capacity & Tiered Retrieval (70/30 Rule) ----
    # 系统级全量检索容量上限（基本事件数）。低于此值全量可见；超过后开启 70/30 分层。
    recall_max_capacity: int = 1000
    # 分层比例：最近 N% 的条目对全局可见，剩余部分仅焦点角色关联可见。
    recall_global_ratio: float = 0.7

    # ---- Recall trace（回忆内容落库，observability / 召回质量回溯） ----
    # 每次 recall 调用把输入与命中的回忆条目写入 ``recall_traces`` 表；
    # 关闭则回忆起只读路径（不新增记录）。只作观测，不影响召回结果。
    recall_trace_enabled: bool = True
    # 单条轨迹最多记录多少个命中条目（防御超大 top_k 撑爆 JSON 列）。
    recall_trace_max_items: int = 60

    # ---- Portrait（人物肖像：per-object 10 级渐进人物摘要 + 记忆疲态）----
    # 等级化：L1 = max(portrait_magic_num, 总长 / growth^(max_levels-1)) —— 取较长，保证最短级(L1)
    #        至少达到魔法数(默认100)、保持有效性；Lk = L1 * growth^(k-1)；
    # 某级预算 > 总长 即停（该级保留原长度=截到总长），最多 portrait_max_levels 级。
    # 压缩率 = 汇总原始字数 / 最长一级字数（模型压缩，忽略/简化/调整）。
    portrait_enabled: bool = True
    portrait_magic_num: int = 100       # L1 基准魔法数
    portrait_max_levels: int = 10
    portrait_growth: float = 2.0        # 每级是上一级倍数
    # 触发：对象「待画像摘要」累计字数 >= context_window / portrait_trigger_divisor 时执行肖像动作。
    portrait_trigger_divisor: int = 600
    # 组合/压缩方式：incremental（已有肖像最长级 + 新摘要 压缩）| full（所有摘要 重建）。
    # "auto"：当累计总长 ≤ context_window * portrait_full_mode_context_ratio（默认 1/6 上下文）时用 full，
    #        否则用 incremental（内容多了就不整包重建）。
    portrait_compose_mode: str = "auto"
    portrait_full_mode_context_ratio: float = 1.0 / 6.0
    # 记忆疲态：分成两个值，**不能共用一个**——
    #   portrait_fatigue：人物肖像摘要，由整体负担 + 对象负担合成（综合）。
    #   event_fatigue：事件，仅由整体记忆负担决定（不看对象负担）。
    # 均默认 1（无遗忘）；<1 时按概率让部分内容概率不可见（期望可见占比 ≈ fatigue），保留偶然性。
    portrait_fatigue: float = 1.0
    event_fatigue: float = 1.0

    # ---- Recall budget management（回忆预算管理，design/1010 §8.5）----
    # 预算 = 本次回忆的总字数，与回忆等级相关。回忆分 recall_level_count 级，各级字数递增
    # (L1=最短=recall_magic_num，之后逐级 ×recall_level_growth，L{count}=最长)。
    # 分配：条目按(评分)排序，贪心——能装最高级就装最高级，装不下下降级，再装不下丢弃；
    # 对话人人物肖像作为回忆条目同场参与。总字数 ≤ recall_budget_chars（0 => physical_redline）。
    recall_budget_enabled: bool = True
    recall_budget_chars: int = 0            # 0 = 用 physical_redline 作预算
    recall_level_count: int = 6
    recall_magic_num: int = 100
    recall_level_growth: float = 2.0
    recall_budget_include_portrait: bool = True
    # 召回时并入相关未完成事件（同一对话人优先；无焦点对象时按查询词重叠）。
    recall_include_unclosed: bool = True
    recall_unclosed_same_object_score: float = 90.0
    # 进入候选的语义、词面门槛，以及相关度视为接近、才允许可及性改序的幅度。
    # 数值只放在配置里，不写进 design/1010。
    recall_semantic_min: float = 0.35
    recall_lexical_min: float = 0.05
    recall_relevance_close: float = 0.08
    # 嵌入模型或维度变更时递增。物理集合名会带上这个版本，旧集合不覆盖。
    recall_embedding_version: int = 1

    # ---- Recall: 并发人物提取与多路检索（本轮新增）----
    # 回忆是否等待回忆前的人物提取结果。默认 False：人物提取并发跑，回忆用即时启发式 focus，不阻塞。
    # True 时回忆前 await，使用完整 focus_role_entries（valence / 焦点档位更准，但更慢）。
    recall_wait_for_role_extraction: bool = False
    # 向量检索单路候选条数（取代历史硬编码 60）。
    recall_n_results: int = 60
    # 是否启用多路召回并集（当前句 / 上下文 / 实体锚定）。关闭时回退到单路「残影+当前输入」。
    recall_multi_route_enabled: bool = True
    # 上下文路中残影参与查询编码的最大字符数（取尾部）；≤0 表示不截断。
    # 防止长残影把当前输入的语义向量稀释。
    recall_query_shadow_cap: int = 1200
    # 检索前 LLM 压缩 query act 文本（对齐入库摘要粒度；短于 min 则跳过）
    recall_query_compress_enabled: bool = True
    recall_query_compress_min_chars: int = 150

    # ---- Ingest LLM session（单次 ingest 内 messages 多轮，Turn2/3 不重复贴全文）----
    # API 无跨请求记忆；Turn2 可与 recall 并发，但 session 写操作须加锁。
    ingest_llm_session_enabled: bool = True
    # 若设置，三 turn 强制同一 model（利于 prefix cache）；None 时各 task 用 task_models。
    ingest_llm_session_task_model: str | None = None

    # ---- Abstraction: Frequent Subset Mining (白皮书 §3.2 唯一触发) ----
    # 每次回忆产生的回忆块 event_id 集合被登记到 ``recall_log``；在集合族中找满足：
    #   - 子集大小 >= abstract_subset_min_size（默认 6；可配置）
    #   - 支持度（跨多少条回忆块被整体覆盖） >= abstract_subset_min_support（默认 12）；连续多条抽象后可抬高至
    #     ``abstract_subset_min_support_escalated``（见 ``abstract_mining_escalate_min_support_after_consecutive_abstracts``）。
    # 挖掘器先枚举频繁闭包；对满足阈值的 **全部**频繁子集（非仅极大）逐个经护栏后合成抽象事件。
    # 合成成功后 ``replace_subset`` 会改写 ``recall_log``；若继续沿用本轮开始时算出的候选与支持度，
    # 可能对「已被替换掉的基本事件」仍尝试合成（过时统计）。默认在每次成功抽象后基于最新日志重新挖掘。
    abstract_mining_max_refresh_rounds: int = 256
    abstract_mining_recent_k: int = 1000
    abstract_mining_min_recent_k: int = 200
    abstract_support_decay_lambda: float = 0.01
    abstract_subset_min_size: int = 6
    abstract_subset_min_support: int = 12
    # 单次 ``mine_and_synthesize`` 内已成功抽象的条数 **>** 该阈值后，后续刷新轮改用更高的最小支持度（默认 >2 即从第 4 条起收紧）。
    abstract_mining_escalate_min_support_after_consecutive_abstracts: int = 2
    abstract_subset_min_support_escalated: int = 13
    # ---- Abstraction narrative coherence gate (自检 → 相关率 a) ----
    # 挖矿子集送入合成前先做 LLM 划分：相干叙事 vs 剔除；相干数低于阈值则跳过本次抽象。
    abstract_narrative_coherence_enabled: bool = True
    abstract_narrative_coherence_min_count: int = 3
    # ---- 抽象覆盖度 → 回忆 RRF 降权（多次被更高阶抽象覆盖则分更低，恒 >0）----
    # 单次新抽象对覆盖集内事件的增量：abstract_coverage_strength / (1 + ln(|C|))，再封顶 abstract_coverage_max。
    abstract_coverage_strength: float = 1.0
    abstract_coverage_max: float = 3.0
    # RRF 融合分乘以 exp(-abstract_coverage_decay_rate * abstract_coverage)；≤0 表示关闭降权。
    abstract_coverage_decay_rate: float = 2.0

    # 未闭合事件总长超过 len_msg * 该比例则触发 80/20 强制分裂（详见 boundary_split_* 设置）。
    # 白皮书 4.2 的物理红线（physical_redline）仍在 ``_check_physical_redline`` 兜底，
    # 但"认知层"的默认路径不再做盲目强制封存——只做评估 → 修复 → 保留 oversized UC。
    # 是否「可疑」由角色抽取与 RoleService 仲裁判断。
    unclosed_force_ratio: float = 1.2
    # ---- 未完成事件封存条件（按单条 UC 字数，非残影总长；design/610）----
    # 字数上限魔法数。单条未完成事件字数 ≥ 该值 → 按原文封存。
    # 匠石可通过环境变量 REMS_UNCLOSED_CHAR_LIMIT 覆盖；0 = 关闭字数触发。
    unclosed_char_limit: int = 2000
    # 闲置封存（相对 last_hit_time / occurred_at）：
    #   (字数 > partial_ratio × unclosed_char_limit 且闲置 ≥ partial_days)
    #   或 (闲置 ≥ hard_days)
    unclosed_idle_seal_enabled: bool = True
    unclosed_idle_partial_days: float = 3.0
    unclosed_idle_hard_days: float = 7.0
    unclosed_idle_partial_ratio: float = 0.5
    # 兼容旧环境变量；逻辑已废弃，勿再依赖。
    unclosed_idle_hours: float = 1.0
    # 立即封存的原文字数门槛（资源目标，不是事件的语义尺度）。
    # 常见工作尺度暂定约 800–1000 字，有弹性，鼓励偏长；本默认取区间中部。
    # 短于该值的已形成事件仍是事件，可在形成窗口多留一些时间等待扩写或结束等待。
    # 0 = 不设门槛，当轮封存。模型不看这个数。可用 REMS_EVENT_MIN_CHARS 覆盖。
    event_min_chars: int = 900
    # 事件工作尺度 ev_len 的上限倍数。模型在 [ev_len, k·ev_len) 内取最接近 ev_len 的闭合。
    # ev_len 使用 event_min_chars。0 表示不设尺度，模型标成事件的都封存。
    event_len_factor: float = 2.0
    # ---- 80/20 Forced Split (2026-05, 白皮书 4.2 升级) ----
    # 评估到 oversized_uc 后，OverlongUCSplitSkill 的目标切分比例与可接受区间。
    # 切点仍由模型基于"逻辑闭环"选择，此处只给数量级指引。
    boundary_split_ratio_target: float = 0.8
    boundary_split_ratio_min: float = 0.7
    boundary_split_ratio_max: float = 0.9
    # 是否启用 OverlongUCSplitRemediator（修复器）。关闭时 oversized UC 将保留为 oversized=True
    # 但不再 force-seal，直接让物理红线层在极端情况下兜底。
    boundary_remediation_enabled: bool = True
    # 硬编码 1：评估只做一层，修复后的输出不再评估。
    boundary_max_remediation_passes: int = 1
    # 是否启用 LLM 评估器（对切分质量做二级评估，成本敏感）。默认关闭。
    boundary_enable_llm_eval: bool = False
    # 回忆时前缀链展开的最大深度——防止"前缀的前缀的前缀……"把回忆块撑爆。
    recall_split_prefix_max_depth: int = 3
    # 递归摘要熔断：某级摘要字符数 **低于** 该阈值则不再生成更高级（白皮书 1.1.3）；按产品约定为 35 字。
    summary_fuse_min_chars: int = 35

    # 情绪唤醒度映射遗忘因子的幂指数：base_forgetting_factor = 100 * arousal ** gamma。
    emotion_arousal_gamma: float = 2.0
    # 事件 activation_energy → 初始遗忘因子：factor = 2 ** (gain * (activation - 0.5))。
    # 中性 0.5 → 1.0；高唤醒 > 1（更难遗忘）；低唤醒 < 1（更快衰减）。测量后校准。
    recall_activation_gain: float = 2.0

    # 白描时间线遗忘半衰期（天）。
    wp_half_life_days: float = 60.0
    
    # 事件静默阈值：当所有角色的有效遗忘因子均低于此值时，事件被设为 SILENT。
    event_silence_threshold: float = 0.02
    forgetting_silence_threshold: float = 0.02

    # ---- Dynamic Recall Compression (白皮书 4.4 扩展与分级压缩) ----
    # 初始向量检索的目标长度倍率（相对于 physical_redline）。
    recall_expansion_factor: float = 2.0
    # 情感精排后的中间过滤目标长度倍率（相对于 physical_redline）。
    recall_intermediate_filter_factor: float = 1.2
    # 初始保持高保真摘要的头部条目比例（由条目数决定）。
    recall_head_ratio: float = 0.66

    # ---- Recall quality: 回忆块相关性抽检 (b) + 动态语义距离钳制 ----
    # 每 N 次 ingest 可选跑一次回忆块抽检；≤0 表示永不跑 **b**（仅 **a** 驱动 EMA，若也未观测则中性）。
    recall_relevance_audit_every_n_ingests: int = 10
    # EMA α；combined = w_coherence·ema_a + (1-w_coherence)·ema_b；**w 为「a」的权重**，默认可取小（权在 **b**）。
    recall_quality_ema_alpha: float = 0.25
    recall_quality_weight_coherence_vs_relevance: float = 0.1
    # True 时对向量检索命中按 distance ≤ cap 预过滤（流 A/B）；False 等同旧行为。
    recall_dynamic_distance_enabled: bool = False
    # 语义距离上限（与本库 VectorStore.distance 同源；通常 1-cos，越小越相似）。combined 高→略抬高 cap。
    recall_dynamic_distance_cap_base: float = 2.0
    recall_dynamic_distance_cap_floor: float = 0.45
    recall_dynamic_distance_cap_ceiling: float = 2.0
    recall_dynamic_distance_sensitivity: float = 0.08
    recall_dynamic_distance_quality_neutral: float = 0.5
    # 将单次 ingest 中流 A+B 向量命中 distance 的全集视作样本，取其分位数（0~100）经 floor/ceiling 夹挤后，
    # 与 EMA-driven 的 semantic_cap 取 **min**（谁更严用谁）；None = 不关分位帽（仅原有 cap 逻辑）。
    recall_dynamic_distance_hit_percentile: float | None = None

    # 兼容旧配置：当前实现已改为始终展开到叶子基本事件并使用 content_raw 作为抽象证据。
    hallucination_anchor_prob: float = 0.3

    # 抽象事件是否额外生成 ``insight``。关闭时抽象合成只产出压缩后的 ``content_raw`` 与可选 decoration；
    # 打开时才要求模型提炼跨事件规律，且该 insight 不是事实摘要本身。
    enable_abstract_insight: bool = False

    # ---- Role capacity & soft-forgetting (白皮书 2.3 容量分配与软遗忘) ----
    # 角色白描容量上限倍率：实际容量 = context_chars / wp_role_capacity_divisor（默认 6.6，即与 len_msg 同阶）。
    # 容量内白描不施加遗忘惩罚；超出后旧条目在 Recall 打分中被遗忘因子惩罚。数据永不物理删除。
    wp_role_capacity_divisor: float = 6.6

    # ---- Dynamic compression ratio control (白皮书 1.2 前置预算计算) ----
    # 事件封存时 sum_len / raw_len 的目标比值，默认 1/6.6。
    compression_target_ratio: float = 0.1515
    # 递归摘要指数衰减因子 (L1 -> L10)；每一级相对于上一级的字数限制比例。
    summary_decay_factor: float = 0.5
    # 角色快照指数衰减因子 (L3 -> L2 -> L1)；L2 = L3 * factor, L1 = L2 * factor。
    snapshot_decay_factor: float = 0.6
    # 缩放因子：在目标压缩率基础上的全局缩放（> 1.0 放松字数，< 1.0 收紧字数）。
    # 既有递归摘要熔断等逻辑保持不变，预算约束通过该因子叠加于其上。
    compression_budget_multiplier: float = 1.0
    # 分项预算占总预算的比例（合计应为 1.0）。
    budget_ratio_summary: float = 0.40     # 默认级摘要
    budget_ratio_snapshot: float = 0.25    # 角色快照（各角色均分）
    budget_ratio_wp: float = 0.25          # 角色白描条目（各角色均分）
    budget_ratio_decoration: float = 0.10  # 装饰
    
    # 是否启用情节装饰（Decoration）：开启后会额外调用一次 LLM 为事件生成感性描述。
    enable_decoration: bool = False

    # 墓碑化时写入 insight 的审计前缀（白皮书 4.3）。
    tombstone_prefix: str = "[TOMBSTONE]"

    # ---- Global pronoun-resolution mode (白皮书 2.2) ----
    # 默认单人模式；生产环境按场景切换为多人。
    user_mode: UserMode = UserMode.SINGLE
    # 单人模式下第一人称与无主语动作默认归属的核心用户 role_id；留空则仍由模型自推。
    core_user_role_id: str | None = None
    # 多人模式下的已知活跃参与者花名册（role_id 或角色名），会被注入 system prompt 做共指消解。
    active_participants: list[str] = Field(default_factory=list)

    # ---- Emotion: EMA (Emotion & Adaptation) dynamic evolution (白皮书 2.5) ----
    # 情绪滚动状态的时间衰减核（小时）：phi(delta) = exp(-lambda * delta_hours)。
    emotion_decay_lambda_per_hour: float = 0.1
    # 情绪滚动状态回溯读取白描尾部的条数。
    ema_history_window: int = 10
    # 事件 activation_energy 由角色滚动 energy 聚合后可选放大。
    activation_energy_gain: float = 1.0

    # ---- Recall intent classification: 软混合权重（本轮新增，去正则硬三选一） ----
    # True（默认）：按词典（可选原型向量）信号算混合系数，对三组 tri_band 权重做凸组合，支持混合意图。
    # False：回退到旧的正则硬选 dominant，便于 A/B。
    recall_intent_soft_blend_enabled: bool = True
    # softmax 温度：越小越「尖」（接近硬选），越大越「平」（更均匀混合）。
    recall_intent_softmax_temperature: float = 1.0
    # fact 基线分：保证无任何信号命中时混合结果回退到 weight_fact。
    recall_intent_fact_baseline: float = 1.0
    # 三类意图词典（子串命中计数，大小写不敏感）。fact 多为残余类，词典可空。
    recall_intent_entity_lexicon: list[str] = Field(
        default_factory=lambda: [
            "谁", "人物", "角色", "轨迹", "做过什么", "干什么", "和谁", "跟谁",
            "where", "who", "character", "person",
        ]
    )
    recall_intent_emotion_lexicon: list[str] = Field(
        default_factory=lambda: [
            "感觉", "心情", "情绪", "害怕", "开心", "难过", "生气", "喜欢", "讨厌",
            "feel", "emotion", "mood", "afraid", "happy", "sad", "angry",
        ]
    )
    recall_intent_fact_lexicon: list[str] = Field(
        default_factory=lambda: [
            "什么时候", "哪里", "为什么", "怎么", "多少", "是不是", "发生了",
            "when", "why", "how", "what", "fact",
        ]
    )
    # 原型向量信号（可选）：用现有嵌入器对每类种子短语预编码，取 cosine 作为附加信号。
    # 默认关闭——哈希嵌入器下原型噪声大；接入真实句向量模型时可开。
    recall_intent_prototype_enabled: bool = False
    recall_intent_prototype_weight: float = 2.0

    # ---- Pluggable recall / RAG profiles (§4.4 extension) ----
    # tri_band | hybrid_literary | personal_log | npc_agent
    recall_profile: str = "tri_band"
    recall_bm25_enabled: bool = False
    recall_bm25_top_k: int = 40
    recall_recency_enabled: bool = False
    recall_recency_window_events: int = 80
    recall_recency_top_k: int = 40
    # RRF channel names for the active profile (empty = backend default).
    recall_rrf_channels: list[str] = Field(default_factory=list)
    # hybrid_literary: ask enrichment for keywords / location (optional JSON fields).
    recall_enrichment_metadata_enabled: bool = False

    # ---- RRF recall modifiers (白皮书 4.4) ----
    recall_rrf_k: int = 60
    recall_factor_alpha: float = 0.5
    recall_mood_beta: float = 0.2
    # 回忆命中强化：温和增长（1.15x），避免指数累积把事件推向"永生"；
    # 上限 3.0 保持强度可区分但有限（原 1.5x/300 会让高频命中事件永不遗忘）。
    recall_reinforce_multiplier: float = 1.15
    recall_forgetting_factor_cap: float = 3.0

    # ---- White-painting collection tier (白皮书 2.3 动态粒度路由·收集端) ----
    # 白描收集时对主要角色（S/A 或单人核心用户）落盘的默认档位字段：l3_decision / l2_interaction / l1_mention。
    wp_primary_field: str = "l3_decision"
    wp_default_field: str = "l2_interaction"
    wp_minor_field: str = "l2_interaction"

    # ---- Performance monitor & load-driven knobs (白皮书 §2.3 高负载自我保护) ----
    # 是否启用非 LLM 主算法的耗时监控。关闭后 timer 退化为 no-op、load_factor 恒为 1.0，
    # 单测与基线性能对比时可关。
    perf_monitor_enabled: bool = True
    # 滚动窗口大小：每个 phase 保留最近 N 次耗时取均值。N 越大对短期尖峰越不敏感。
    perf_monitor_window_size: int = 32
    # 各 phase 的健康容忍上限（毫秒）。phase 的滚动均值越过此值即开始为 load_factor 贡献。
    # 数值越大 = 对长耗时的容忍度越高 = 系统更晚进入"高负载自我保护"。
    # 当前监控的非 LLM phase（按需扩展，不在表里的 phase 第一次 record 时按默认 200ms 注册）：
    #   - rag_search          ：向量库语义检索
    #   - recall_assembly     ：回忆块组装
    #   - abstraction_mining  ：频繁子集挖掘
    #   - narrative_dedupe    ：抽象事件叙事线判重
    perf_phase_tolerance_ms: dict[str, float] = Field(
        default_factory=lambda: {
            "rag_search": 200.0,
            "recall_assembly": 500.0,
            "abstraction_mining": 2000.0,
            "narrative_dedupe": 300.0,
        }
    )
    # load_factor 的硬上限，避免极端尖峰把下游阈值放大到无意义的程度。
    perf_load_factor_max: float = 4.0
    # Deprecated: 2026.06 双通道 PerfMonitor 不再用 load_factor 抬高静默阈值。
    forgetting_overload_silence_boost: float = 4.0
    perf_recall_phases: list[str] = Field(
        default_factory=lambda: ["rag_search", "recall_assembly"],
    )
    perf_abstract_phases: list[str] = Field(
        default_factory=lambda: ["abstraction_mining", "narrative_dedupe"],
    )

    # ---- Narrative-line dedupe for abstract events (覆盖 is_fired / replace_subset 之间的灰区) ----
    # 默认关闭。开启后：已存在的"完全包含"去重仍在 replace_subset 里完成；本项专门拦
    # "几乎相同的叙事线再次浮现"——overlap 高 + novelty 不足时跳过新合成。
    enable_narrative_dedup: bool = False

    # 阈值含义：overlap = |S ∩ A.leaves| / |S|；novelty = |S \ A.leaves| / |S|
    narrative_dup_overlap_threshold: float = 0.9
    narrative_dup_novelty_min_ratio: float = 0.1
    # 绝对量护栏：novelty * |S| 必须 >= 这个绝对值才视为"显著新增"。
    # 防止短子集靠"差 1 个事件"的比例诡计骗过 overlap 阈值。
    narrative_dup_novelty_min_abs: int = 2
    # 候选 × 已有抽象事件的全比成本：默认对最近 K 条已合成抽象做对比。
    # PerfMonitor 检测到过载时会按 load_factor 自动收紧到 narrative_dup_min_compare_recent_k。
    narrative_dup_compare_recent_k: int = 200
    narrative_dup_min_compare_recent_k: int = 20

    @property
    def context_chars(self) -> int:
        """Effective character budget derived from token window.

        用 ``context_window * chars_per_token`` 估算整段上下文可承载的字符上限，作为后续 1/66、10/66 比例约束的基数。
        """
        return int(self.context_window * self.chars_per_token)

    @property
    def len_msg(self) -> int:
        """1/66 of context — max single input / event length (chars).

        单条消息或单事件 L0 的推荐/硬上限比例：约为 ``context_chars`` 的 1/66（白皮书 1.1.7 物理防御与4.2 触发语义）。
        """
        return int(self.context_chars / 66)

    @property
    def wp_role_capacity(self) -> int:
        """Per-role white-painting capacity in characters (白皮书 2.3).

        角色白描时间线在「全保真无遗忘」模式下可承载的最大字符总量。
        默认等于 ``context_chars / 6.6``（与 ``len_msg`` 同阶）。
        容量内不施加遗忘惩罚；超出后旧条目仅在 Recall 打分中被遗忘因子惩罚，数据永不删除。
        """
        return int(self.context_chars / self.wp_role_capacity_divisor)

    @property
    def physical_redline(self) -> int:
        """10/66 of context — absolute physical safety ceiling (chars).

        回忆块等组装内容的物理安全上限：约为 ``context_chars`` 的 10/66，对应白皮书 4.4 所述约 1/6.6「主权约束」，
        用于防止单次上下文爆炸。
        """
        return int(self.context_chars * 10 / 66)

    @property
    def safe_watermark(self) -> int:
        """2/66 of context — target watermark after forced compaction (chars).

        触发物理红线强制整理后，残影内容希望保留的尾部目标长度（约 2/66上下文），用于代谢模块截断策略。
        """
        return int(self.context_chars * 2 / 66)
