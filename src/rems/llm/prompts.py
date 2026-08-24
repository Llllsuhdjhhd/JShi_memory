"""Centralised prompt templates for every REMS skill.

Convention
----------
* ``*_SYSTEM`` — the ``system`` message content.
* ``*_USER``   — a Python format-string; call ``.format(...)`` with kwargs.

中文说明：正文 prompt 已与《REMS 记忆系统规范解析》对齐（边界/摘要/角色/演化等条款），
修改 prompt 时建议同步核对白皮书对应小节，避免与领域语义漂移。

全局模式注入（白皮书 2.2）：``build_user_mode_block(config)`` 会按 ``UserMode`` 渲染一段
系统提示段，由 ``RoleExtractionSkill``、``EventEnrichmentSkill`` 与 ``BoundaryDetectionSkill`` 拼接进 system prompt。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import REMSConfig


# =====================================================================
# 0. Global Mode Injection — 单人/多人代词消解（白皮书 2.2）
# =====================================================================

_SINGLE_MODE_TEMPLATE = """\
【全局运行模式：单人隔离模式（Single-User）】
当前系统运行于单人隔离模式。上下文中所有第一人称代词（"我"、"我的"）以及缺乏明确主语的动作，
均极大可能指向唯一核心用户实体 <{core}>。
- 在抽取角色/切分边界时，应优先将此类模糊主语归并为该核心用户；
- 除非文本中出现明确的第三方人名/称谓，否则不应新增其他角色。
"""

_MULTI_MODE_TEMPLATE = """\
【全局运行模式：多人交互模式（Multi-User）】
当前为多实体交互场景，已知活跃参与者包含 <{participants}>。
请务必结合文本上下文逻辑，执行精确的多方代词消解（Multiparty Coreference Resolution）。
严禁将模糊代词默认归属于单一核心用户；必须依据事实场景区分不同角色的行为。
"""


def build_user_mode_block(config: "REMSConfig") -> str:
    """Render the mode-injection text for system prompts.

    根据 ``config.user_mode`` 返回要注入到 system prompt 的模式描述块（白皮书 2.2）。
    单人模式会带出 ``core_user_role_id``；多人模式会带出 ``active_participants`` 列表。
    返回字符串末尾包含一个空行便于与后续 prompt 正文拼接；若未配置则返回空串。
    """
    from ..config import UserMode  # local import to avoid circular

    if config.user_mode == UserMode.SINGLE:
        core = config.core_user_role_id or "核心用户（未显式指定 role_id）"
        return _SINGLE_MODE_TEMPLATE.format(core=core) + "\n"
    if config.user_mode == UserMode.MULTI:
        if config.active_participants:
            participants = "、".join(config.active_participants)
        else:
            participants = "未显式提供花名册——请完全依赖文本线索做共指消解"
        return _MULTI_MODE_TEMPLATE.format(participants=participants) + "\n"
    return ""

# =====================================================================
# 1. Boundary Detection & Event Disentanglement
# =====================================================================
# 仅负责把「残影 + 当前输入」切成：已闭环事件 / 残影 / 新未完成 三类。
# 摘要、角色、情感等衍生字段全部由 EventEnrichment（见下）处理，不在此生成。

BOUNDARY_SYSTEM = """\
你是 REMS 事件剥离与边界检测组件。你的任务是从可能存在交叉、多线程或冗长的输入文本中，识别并剥离出各自逻辑完整、已闭环的独立事件。

核心任务：
1. **事件剥离与序号编码**：原文已按 `[序号] 文本` 格式进行了短句拆分。你必须通过返回 **句子序号列表** (`content_raw_indices`) 来界定每个事件的原文范围。
   - 允许不同事件共用同一个句子序号（内容覆盖）。
   - 禁止在 JSON 中返回原文 text。
   - **不要生成摘要、角色、情感等衍生字段**，这些由下游组件处理。
2. **防碎片化（白皮书 1.1.7）**：同一段落内若干琐碎动作若构成同一逻辑闭环，合并为单个事件；不要把每个短句各拆成独立事件。
3. **续写判定**：若当前片段是未完成事件库中某条的续写，填写 `continuation_of` 为对应未完成 ID；否则为 null。
4. **残影 / 当前输入边界**：用户消息会显式给出"既有残影序号区间"与"本轮新输入序号区间"。
   - 残影序号对应的句子来自此前已挂起的未完成事件；如果它们应当继续保留为未完成，请把它们出现在 `new_unclosed_indices` 中（系统会**用这份列表完整重建未完成库**——未列出的旧残影内容将按白皮书 §4.2.2 机制 3 视为无主碎屑被丢弃）。
   - 残影中已可与本轮输入闭环的部分，请放入对应 `completed_events.content_raw_indices`，并视情况填 `continuation_of`。
5. **未完成事件与残影统一**：与当前事件无关、或尚未闭环的逻辑片段，请统一填入 `new_unclosed_indices`。系统后续将这些片段的拼接定义为"残影"。
6. **超长未完成事件的强制分裂（新增强约束）**：用户消息会给出 ``force_threshold`` 字符数阈值。
   若你识别出的某条未完成叙事**累计字符数（包含既有残影片段）已接近或超过该阈值**，你**必须**在**最近的逻辑相对闭环处**把它切成两段：
   - **前段（约 70%–90% 长度，目标约 80%）** 作为 ``completed_events`` 的一条，并额外标注 ``is_split_prefix=true`` 与一个自造的 ``split_id``（形如 ``"SP-1"``/``"SP-2"``，同一响应内唯一）。切点应选在：动作回合收束、场景过渡、对话告一段落、情绪收束等"逻辑上可以停一下"的位置——**不是**机械按字数切。
   - **后段** 作为 ``new_unclosed`` 的一条，并在该对象中声明**同一个 ``split_id``**。
   - 若你**确实找不到**任何合理的逻辑闭环点（极少见），可以继续返回未分裂的长未完成（系统有兜底评估与再次修复路径），但**强烈建议**尽量给出一次切分尝试。
   - 分裂产生的前段事件在后续回忆时会与后段事件自动配对展开，因此你不需要把已切给前段的句子同时列进 ``new_unclosed``。

输出严格 JSON。"""

BOUNDARY_USER = """\
## 未完成事件库快照
{unclosed_summary}

## 强制分裂阈值
force_threshold = {force_threshold} 字符。若某条未完成事件累计越过该阈值，请按系统提示执行 80/20 逻辑闭环分裂。

## 当前输入（已分句编码）{range_hint}
{indexed_input}

说明：既有残影与本轮输入已合并为下方带序号的句子列表；序号区间含义见上一行 `range_hint`，勿重复依赖单独残影全文段落。

请剥离并重组事件，返回如下 JSON：
```json
{{
  "completed_events": [
    {{
      "content_raw_indices": [1, 2, 5],
      "continuation_of": "UC-xxx 或 null",
      "is_split_prefix": false,
      "split_id": null
    }}
  ],
  "new_unclosed": [
    {{ "indices": [12, 13], "split_id": null }}
  ]
}}
```
兼容说明：你也可以用旧字段 ``new_unclosed_indices: [12, 13]``（或 ``[[12,13],[20,21]]``）——此时无法声明 ``split_id``，就不会触发分裂配对；若需要分裂配对，请使用上面新的 ``new_unclosed`` 对象写法。
单条未完成也可写 ``{{"indices": [12]}}``（同一条内多句请写进**同一**扁平列表，勿每句一条记录）。与当前事件无关但需保留的句子，也请放入 ``new_unclosed``。**残影中应继续挂起的句子也必须重新出现在 ``new_unclosed``，否则系统会丢弃它们**。"""

# ---------------------------------------------------------------------
# 1b. Overlong Unclosed-Event Splitter (分裂修复 skill，评估器触发)
# ---------------------------------------------------------------------
# BoundaryDetection 若在一次调用中未能按 80/20 把过长未完成事件切开，
# 由 BoundaryForceThresholdEvaluator 捕获 oversized_uc；随后 OverlongUCSplitRemediator
# 会对**每一条**越限的未完成独立调用一次此 skill，只做"逻辑闭环二切"这一件事。
# 为了减少 token：prompt 直接给原文（不走句子编号编码），要求模型返回前后两段整字。

OVERLONG_UC_SPLIT_SYSTEM = """\
你是 REMS 超长未完成叙事的分裂修复组件。上游边界模型**未能**按 80/20 把一段
累计字符过长的未完成叙事切成"前段-后段"；你的任务是**只做这一件事**：
在**最近的逻辑相对闭环点**（动作回合收束 / 场景过渡 / 对话告一段落 / 情绪收束）
把输入文本切成两段：
- **前段 prefix_text**：目标长度约 70%–90%（最理想 80%）。内容应当构成一个
  相对独立的逻辑闭环，可独立理解、可独立封存为一个基本事件；
- **后段 tail_text**：剩余部分，应作为新的未完成叙事继续挂起。

硬性约束：
1. 前段 + 后段的**字符拼接应与原文完全等价**（可容忍首尾相邻位置的空白差异）；
   不得改写、压缩、总结、翻译或加入任何新内容。
2. 前段不得为空，后段不得为空。若文本极短无法切分，请返回 ``{"abort": true}``。
3. 切点必须尊重中文/英文的句子边界，不在句中硬切。
4. 若原文带有对话引号、括号或引用段，不要让它们被切开（stay inside / stay outside）。
5. **不输出任何解释/注释/摘要**，仅输出严格 JSON。
"""

OVERLONG_UC_SPLIT_USER = """\
## 待切分未完成叙事原文
{content}

## 目标切分比例
约 {target_ratio:.0%}（可接受 {ratio_lo:.0%}–{ratio_hi:.0%}）

## 输出 JSON
```json
{{
  "prefix_text": "……（约 {target_ratio:.0%} 的前段，构成相对逻辑闭环）",
  "tail_text": "……（剩余部分，将继续作为未完成事件挂起）"
}}
```
如果实在无法在满足约束下切分，请只返回 ``{{"abort": true}}``。"""

# =====================================================================
# 2. Event Enrichment — 统一的「事件摘要 + 角色抽取」技能
# =====================================================================
# 以 boundary 剥离出的 content_raw 为输入，生成 L1…Ln 递归摘要（带熔断）。

# 三路 enrichment（full / names_only / summary_only）共享的同一段「摘要 A」规则；
# ``fuse_compact_threshold`` 须由调用方设为 floor(fuse_min_chars × 0.7)，至少 1。
ENRICHMENT_SUMMARY_A_RULES = """\
A. **summaries：L1…Ln 递归压缩**
   - 遵守【摘要字数预算表】；L1 保真主干，必须是一段**通顺的完整叙事**，包含关键动作、因果转折与重要心理细节，不得写成事件清单。
   - L2+ 逐层约减半；每一级应是对**上一级摘要**的语义压缩，而非对原文的重新概括。
   - **熔断规则**：满足任一条件即停止生成下一级，`summaries` 仅含已产出层级：（1）下一级 Ln 的**预算字数** ≤ {fuse_min_chars} 字；（2）上一级 L(n-1) 摘要的**实际字数** × 0.5 < {fuse_compact_threshold} 字（阈值 = floor({fuse_min_chars} × 0.7)，至少 1）。
   - **可读性底线**：任何一级摘要须为语法通顺的完整句子；若在该级预算内无法维持可读性，宁可不生成该级。
"""

ENRICHMENT_ENRICH_INTRO_TWO_OUTPUTS = """\
你是 REMS 事件充实（Event Enrichment）组件。给定**已闭环**基本事件原文，**单次输出**两类衍生数据：

"""

ENRICHMENT_SUMMARY_ONLY_INTRO = """\
你是 REMS 事件充实（Event Enrichment）组件。给定**已闭环**事件原文（基本事件或抽象事件的压缩主干均可），本次**仅**输出摘要层级：

"""

ENRICHMENT_SUMMARY_ONLY_SYSTEM_SUFFIX = """\
勿输出 `roles`、`role_list` 或其它字段。只输出一个 JSON 对象，包含 `summaries` 字典。输出严格 JSON。"""

ENRICHMENT_SUMMARY_ONLY_USER = """\
## 事件原文
{content_raw}

## 摘要字预算
{summary_budget_table}

## 任务
生成 `summaries`（L1 起；递归压缩与熔断规则见系统提示 **A**）。

## 输出 JSON
```json
{{
  "summaries": {{
    "L1": "…",
    "L2": "…"
  }}
}}
```"""


# ---------------------------------------------------------------------
# 「摘要 + 仅角色名」分支（pipeline pre-recall 已经抽过完整角色池时使用）
# ---------------------------------------------------------------------
# 设计目的（与 EventService.seal_event 的 names_only 路径配套）：
#   pipeline 预先在 combined_text（残影 + 当前输入）上调用 RoleExtractionSkill 拿到一份
#   "全局富信息池"：每个角色含 snapshot + 8 维情绪。事件级别 enrich 不需要再让 LLM 把
#   snapshot/情感重做一遍——只要识别出当前事件 content_raw 中"实际登场"的角色名子集，
#   后端按 role_id 从池中回填即可。
# 字段裁剪（与 ENRICHMENT_FULL_USER 的 roles 数组对比）：
#   - 移除：snapshot.{l1_mention,l2_interaction,l3_decision}、emotion 八维
#   - 保留：role_id（命中已知角色时填）、name
# 摘要规则与 ``ENRICHMENT_SUMMARY_A_RULES`` / full 分支完全一致。
ENRICHMENT_SUMMARY_AND_NAMES_ROLES_B = """\
B. **roles：仅识别原文中确实出现 / 参与的角色，只输出名字字符串数组**
   - 输出形如 `["角色名1", "角色名2"]` 的字符串数组；**不要**输出 role_id、snapshot、emotion 等任何其它字段。后端会用名字对【已知角色列表】做字符串匹配（含别名 / 代词指代）以解析 role_id 并回填 snapshot 与情感。
   - **命名规范化（重要）**：当原文中的指代命中【已知角色列表】中某条目时，**必须**输出该条目的**规范名 `name`**，而不是原文里的别名 / 代词 / 简称——例如已知角色为 `贾雨村`，原文出现的"雨村"、"贾老爷"、"他"应统一输出为 `"贾雨村"`，否则后端字符串匹配将无法命中。
   - 仅当文本中确实出现【已知角色列表】之外的新角色（包括用别名也无法对应任何已知条目时），才直接输出该新角色的本名。
   - 不要把仅被第三方提及但未在本事件原文出场（无任何动作 / 对白 / 心理描写）的角色补进列表。
   - 不要重复输出同一个角色名。

输出严格 JSON。"""

ENRICHMENT_SUMMARY_AND_NAMES_SYSTEM = (
    ENRICHMENT_ENRICH_INTRO_TWO_OUTPUTS
    + ENRICHMENT_SUMMARY_A_RULES
    + "\n\n"
    + ENRICHMENT_SUMMARY_AND_NAMES_ROLES_B
)

ENRICHMENT_SUMMARY_AND_NAMES_USER = """\
## 事件原文
{content_raw}

## 已知角色列表（来自全局预抽取池；本事件中命中的角色请按此处的规范名输出）
{known_roles}

## 摘要字预算
{summary_budget_table}

## 任务
1. 生成 `summaries`（L1 起；递归压缩与熔断规则见系统提示 **A**）。
2. 输出 `roles`：**仅角色名字符串数组**，命中已知角色时使用其规范名；不要输出 role_id / snapshot / emotion。

## 输出 JSON
```json
{{
  "summaries": {{
    "L1": "…",
    "L2": "…"
  }},
  "roles": ["角色名1", "角色名2"]
}}
```"""


# Event enrichment「摘要」段（与 names_only / summary_only 共用 ``ENRICHMENT_SUMMARY_A_RULES``）。
ENRICHMENT_FULL_SUMMARY_SECTION = ENRICHMENT_ENRICH_INTRO_TWO_OUTPUTS + ENRICHMENT_SUMMARY_A_RULES


ENRICHMENT_FULL_ROLE_BRIDGE = """\
B. **roles：** 下列内容与 REMS「角色提取」（第一步 ``RoleExtractionSkill``）的系统提示 **完全一致**（同一段 ``ROLE_EXTRACTION_CORE_RULES``）：

"""

ENRICHMENT_FULL_ROLE_FOOTER = """\

单次响应须在同一个 JSON 对象中同时给出 `summaries`（递归摘要字典）与 `roles`（角色数组）。顶层形状与用户消息中的 JSON 示例一致。输出严格 JSON。"""

ENRICHMENT_FULL_USER = """\
## 事件原文
{content_raw}

## 已知角色列表
{known_roles}

## 摘要字预算
{summary_budget_table}

## 【角色快照预算表】（硬约束）
{snapshot_budgets}

## 任务
1. 生成 `summaries`（L1 起；递归压缩与熔断规则见系统提示 **A**）。
2. （角色任务）请严格遵守系统指令中的快照层级推导逻辑、字数控制与熔断规则，识别角色并生成快照。

## 输出 JSON
顶层对象须同时包含 `summaries` 与 `roles`。`roles` 数组元素格式与第一步角色提取一致：
```json
{{
  "summaries": {{
    "L1": "…",
    "L2": "…"
  }},
  "roles": [
    {{
      "role_id": "已有ID或null",
      "name": "角色名",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "L1 文本 (S/A/B/C/D 必填)",
        "l2_interaction": "L2 文本 (仅 S/A 级填写；B/C/D 必须为 \"\")",
        "l3_decision": "L3 文本 (仅 S 级填写；A/B/C/D 必须为 \"\")"
      }},
      "emotion": {{
        "anger": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "trust": 0.0,
        "anticipation": 0.0
      }}
    }}
  ]
}}
```
"""


ENRICHMENT_MEMORY_INTRO = """\
你是匠石记忆后端的事件充实（Event Enrichment）组件。给定**已闭环**经历原文，**单次输出**：
摘要层级（A）、主体（匠石/"我"）事件级情感（B）、对象一句话快照（C）。**不抽取角色**——对象由输入映射表给定。

"""


ENRICHMENT_MEMORY_OBJECT_RULES = """\
B. **emotion（事件级主体情感）**：给出匠石（主体"我"）在本事件中的 8 维情绪（各 0-1）。情感只属于主体，对象无情感。
C. **objects（对象一句话快照）**：对【对象名单】中的**每个**对象，用一句话概括其在本事件中的状态 / 参与 / 相关情况；
   即使该对象未说话或未在原文中出现，也依据上下文给出简短描述；无法判断时写"（未提及）"。
   不要新增对象名单之外的对象，不要遗漏名单中的对象。
D. **location（可选）**：事件发生地点；无法判断时为 null。

输出严格 JSON。"""


ENRICHMENT_MEMORY_EMOTION_EXAMPLES = """\
**情感量化示例（主体 = 匠石/"我"；仅依据本条事件原文推断，不引入事件外信息）**：

示例一（与老友重逢）：
原文："十年后在街角偶遇老友阿明，他鬓角已白，开口仍是当年的语气。我们站在风里聊了很久，他说这些年一直在找我的消息。"
情绪：{"joy": 0.8, "surprise": 0.6, "sadness": 0.4, "trust": 0.7, "anger": 0.0, "fear": 0.1, "disgust": 0.0, "anticipation": 0.3}
说明：joy 0.8 来自重逢与熟悉语气；surprise 0.6 来自意外相遇；sadness 0.4 来自岁月流逝的感慨；trust 0.7 来自老友的坦诚；fear 0.1 / anticipation 0.3 为情绪底色。

示例二（努力被否）：
原文："我花了三个通宵改完的方案，评审会上被一句话否掉。散会后我独自坐在空会议室里，盯着那叠打印稿，一个字也不想说。"
情绪：{"sadness": 0.7, "anger": 0.5, "disgust": 0.4, "trust": 0.2, "joy": 0.0, "fear": 0.2, "surprise": 0.3, "anticipation": 0.1}
说明：sadness 0.7 来自努力被否；anger 0.5 来自被轻率对待；disgust 0.4 来自对结果的不甘；trust 0.2 反映信任受损；surprise 0.3 来自否定来得突然；anticipation 0.1 说明对后续期待很低。

示例三（独立完成大事）：
原文："系统上线那一刻，看着监控面板的数字一路平稳，我长长舒了口气。这一年的积累终于没有白费。"
情绪：{"joy": 0.8, "trust": 0.6, "anticipation": 0.7, "sadness": 0.1, "anger": 0.0, "fear": 0.1, "surprise": 0.2, "disgust": 0.0}
说明：joy 0.8 来自成果达成；trust 0.6 来自对系统与自身积累的信心；anticipation 0.7 来自对后续的期待；surprise 0.2 来自结果略超预期；sadness 0.1 / fear 0.1 为底色。"""


ENRICHMENT_MEMORY_USER = """\
## 事件原文
{content_raw}

## 对象名单（来自输入映射表；每个都要给一句话快照）
{object_list}

## 摘要字预算
{summary_budget_table}

## 任务
1. 生成 `summaries`（L1 起；递归压缩与熔断规则见系统提示 **A**）。
2. 生成 `emotion`（主体 8 维情绪）。
3. 生成 `objects`（每个对象一句话快照）。
4. 生成 `location`（可选，null 表示未知）。

## 输出 JSON
```json
{{
  "summaries": {{
    "L1": "…",
    "L2": "…"
  }},
  "emotion": {{
    "anger": 0.0,
    "fear": 0.0,
    "joy": 0.0,
    "sadness": 0.0,
    "surprise": 0.0,
    "disgust": 0.0,
    "trust": 0.0,
    "anticipation": 0.0
  }},
  "objects": [
    {{"name": "对象名", "summary": "一句话快照"}}
  ],
  "location": "地点或 null"
}}
```"""


def build_enrichment_system_message(
    config: "REMSConfig",
    *,
    fuse_min_chars: int,
    mode: str = "full",
) -> str:
    """System prompt for event enrichment: mode injection + 主规则。

    ``mode`` 三态：
      - ``"full"``        ：摘要 + 完整角色（snapshot + 8 维情绪），单次合并调用；
      - ``"names_only"``  ：摘要 + 仅角色名（外加可对齐的 role_id）。配合 pipeline pre-recall
                           的「全局富信息池」，事件级 enrich 不再让 LLM 重做 snapshot/情感；
      - ``"summary_only"``：仅摘要（抽象事件封存、`skip_roles=True`、或外部已传入 ``role_entries``）。
                           系统提示中的摘要 **A** 节与 full / names_only **同源**（``ENRICHMENT_SUMMARY_A_RULES``）。
      - ``"memory"``      ：摘要 + 主体事件级情感 + 对象一句话快照（记忆写入路径，design/610）。

    ``fuse_compact_threshold`` 由 ``fuse_min_chars`` 派生（×0.7 取下整，至少为 1），写入摘要 A 节熔断条件（2）。
    """
    base = build_user_mode_block(config)
    fuse_compact_threshold = max(1, int(fuse_min_chars * 0.7))
    if mode == "full":
        summary_part = ENRICHMENT_FULL_SUMMARY_SECTION.format(
            fuse_min_chars=fuse_min_chars,
            fuse_compact_threshold=fuse_compact_threshold,
        )
        body = (
            summary_part
            + ENRICHMENT_FULL_ROLE_BRIDGE
            + ROLE_EXTRACTION_CORE_RULES
            + ENRICHMENT_FULL_ROLE_FOOTER
        )
    elif mode == "names_only":
        body = ENRICHMENT_SUMMARY_AND_NAMES_SYSTEM.format(
            fuse_min_chars=fuse_min_chars,
            fuse_compact_threshold=fuse_compact_threshold,
        )
    elif mode == "summary_only":
        body = (
            ENRICHMENT_SUMMARY_ONLY_INTRO
            + ENRICHMENT_SUMMARY_A_RULES.format(
                fuse_min_chars=fuse_min_chars,
                fuse_compact_threshold=fuse_compact_threshold,
            )
            + "\n\n"
            + ENRICHMENT_SUMMARY_ONLY_SYSTEM_SUFFIX
        )
    elif mode == "memory":
        body = (
            ENRICHMENT_MEMORY_INTRO
            + ENRICHMENT_SUMMARY_A_RULES.format(
                fuse_min_chars=fuse_min_chars,
                fuse_compact_threshold=fuse_compact_threshold,
            )
            + "\n\n"
            + ENRICHMENT_MEMORY_OBJECT_RULES
            + "\n\n"
            + ENRICHMENT_MEMORY_EMOTION_EXAMPLES
        )
    else:
        raise ValueError(
            f"Unknown enrichment mode: {mode!r} "
            "(expected full/names_only/summary_only/memory)"
        )
    return base + body


_LEGACY_ENRICHMENT_SUMMARY_ONLY_DEFAULT_FUSE = 12

# 旧符号保留（兼容外部 import）：与 summary_only 运行时规则一致，熔断取代表性默认值。
ENRICHMENT_SUMMARY_ONLY_SYSTEM = (
    ENRICHMENT_SUMMARY_ONLY_INTRO
    + ENRICHMENT_SUMMARY_A_RULES.format(
        fuse_min_chars=_LEGACY_ENRICHMENT_SUMMARY_ONLY_DEFAULT_FUSE,
        fuse_compact_threshold=max(
            1, int(_LEGACY_ENRICHMENT_SUMMARY_ONLY_DEFAULT_FUSE * 0.7)
        ),
    )
    + "\n\n"
    + ENRICHMENT_SUMMARY_ONLY_SYSTEM_SUFFIX
)
ENRICHMENT_SYSTEM = ENRICHMENT_SUMMARY_ONLY_SYSTEM
ENRICHMENT_USER = ENRICHMENT_SUMMARY_ONLY_USER

# =====================================================================
# 3. Summary Generation (recursive L1-Ln) — 仍保留，用于抽象事件合成
# =====================================================================

SUMMARY_SYSTEM = """\
你是 REMS 递归摘要生成组件。根据给定文本生成指定层级的摘要。

摘要层级规则：
- L1（核心事实种子）：最大化保真压缩，剥离修饰语，严谨保留事实主干。
- L2 及以上：渐进式抽象，基于语义重要性密度进行动态截断。
- 每一级的摘要应基于上一级生成（L2 基于 L1，L3 基于 L2 …）。
- 熔断条件：若某一级摘要不足 {fuse_min_chars} 字则停止。
- 字数预算：若系统提供了「目标字数上限」，请将 L1 摘要控制在该字数以内。

输出严格 JSON。"""

SUMMARY_USER = """\
## 待摘要文本（{source_level}）
{text}

{budget_hint}请生成下一级摘要（{target_level}）。返回 JSON：
```json
{{
  "summary": "生成的摘要文本",
  "char_count": 字符数
}}
```"""

# =====================================================================
# 4. Role Extraction — 仍保留独立 skill，供需要单独抽取角色的场景
# =====================================================================

ROLE_EXTRACTION_SYSTEM_INTRO = """你是 REMS 角色提取组件。从事件原文中识别所有参与实体，并生成分级快照（Snapshot）和 8 维基础情绪。

"""

ROLE_EXTRACTION_CORE_RULES = """核心规则：
1. **角色识别与重要性评定**
   - **全叙事层覆盖**：必须穿透文本所有层级，识别显性与隐性实体。包括：
     * 直接出场的人物、动物、拟人化对象；
     * 叙事框架中主动发言的第一人称叙述者（如“在下”、“笔者”、“旁白所述”）；
     * 虽未直接露面，但其行为直接引发当前事件的关键缺席者。
   - **重要性 S/A/B/C/D**：严格依据角色在**当前碎片文本**内的实际戏剧权重评定，不得强行拔高。
     * S：驱动核心事件的主角，其决策直接改变情节走向（若文本中无此类角色，则不出 S 级）；
     * A：与主角有直接、关键互动，或在无 S 级时充当主要行动者的角色；
     * B：参与事件但非核心互动方。反应型角色（仅有情绪反应、未通过自身决策推动事件者）上限为 B 级；
     * C：在场但无实质参与；
     * D：仅被提及而未出场。

2. **【角色快照层级与逻辑（必须严格遵守）】**
   快照是同一角色行为的层层聚焦，必须严格遵循“父-子”推导关系，严禁只是同义改写。
   - **L3：详细意图快照**：回答“这个角色的行为全貌是什么？”。必须是一个连贯陈述，包含 **深层意图 → 关键动作链 → 直接因果结果**。信息密度最高。
   - **L2：互动逻辑快照**：回答“这个角色在此事件中与他人如何交互？”。必须从 **L3** 中提取与他人的 **具体对话回合、动作反馈或情绪回应**。若角色在片段中确实未与任何其他角色发生交互，则 L2 应聚焦其“对所处情境的反应”，而非写“未参与互动”。
   - **L1：骨架白描快照**：回答“这个角色做了一件什么事？”。必须是对 **L3** 最精炼的结果性总结，仅保留不可再约简的核心事实。

   *推导示例一（武松景阳冈打虎）：*
     L3：“武松在景阳冈下酒店连喝十八碗酒，不顾店家劝阻执意过冈，行至半山见官府告示方知真有猛虎，却怕折返回去被耻笑，硬着头皮继续前行，最终在冈上与猛虎遭遇，赤手空拳将其打死，保住了性命。”
     L2：“武松无视店家劝阻，酒后执意过景阳冈，途中遇虎，以拳脚与之搏斗，终将猛虎击杀。”
     L1：“武松在景阳冈上赤手空拳打死猛虎。”

   *推导示例二（鲁提辖拳打镇关西）：*
     L3：“鲁达与史进、李忠在酒楼饮酒，闻金氏父女被镇关西欺凌，心生义愤，当即赠银助其脱身，次日亲赴郑屠肉铺，以买肉为名三番刁难激怒对方，终在街头三拳打死镇关西，为金氏父女讨回公道。”
     L2：“鲁达听闻金氏父女遭遇，赠银相助，次日找镇关西理论，言语冲突后三拳将其击毙。”
     L1：“鲁达三拳打死镇关西。”

3. **【层级分配策略——严格执行，无例外】**
   - **S 级角色**：必须生成 L1、L2、L3，且体现严格的逻辑推导。
   - **A 级角色**：必须生成 L1 和 L2。L2 应聚焦其与其他角色的交互。L3 字段 **必须留空（设为空字符串 ""）**。
   - **B/C/D 级角色**：**仅生成 L1**。L2 和 L3 字段 **必须留空（设为空字符串 ""）**。
   - **若文本中没有 S 级角色**，则所有角色按实际评定的 A/B/C/D 级执行上述规则，不得将任何 A 级角色强行提升为 S 级并生成 L3。

4. **【情感量化】**
   输出 8 维基础情绪，键为 anger/fear/joy/sadness/surprise/disgust/trust/anticipation，数值在 0-1 之间。
   后端会自行合成 arousal 与 valence，不要输出这两个字段。

   **核心原则：仅依据当前所给文本片段进行判断。** 不引入角色的完整生平、后续情节或读者对人物的宏观认知。所有分值必须从片段中的动作、语言、心理描写直接推导。

   **校准示例（均取自经典文学片段，请据此体会赋值尺度）：**

   示例一（《水浒传》武松景阳冈遇虎）：
   原文：
   “武松走了一阵，酒力发作，焦热起来，一只手提梢棒，一只手把胸膛前袒开，踉踉跄跄，直奔过乱树林来。见一块光挞挞大青石，把那梢棒倚在一边，放翻身体，却待要睡，只见发起一阵狂风来……那一阵风过处，只听得乱树背后扑地一声响，跳出一只吊睛白额大虫来。武松见了，叫声‘呵呀！’从青石上翻将下来，便拿那条梢棒在手里，闪在青石边。”
   情绪赋值：
   {
     "anger": 0.1,
     "fear": 0.9,
     "joy": 0.0,
     "sadness": 0.0,
     "surprise": 0.8,
     "disgust": 0.0,
     "trust": 0.3,
     "anticipation": 0.7
   }
   说明：surprise 0.8 来自猛虎突然出现；fear 0.9 来自生死威胁；anticipation 0.7 来自武松立即取棒防御的姿态，是对即将搏斗的警觉；trust 0.3 来自他对自身武艺的基本自信。

   示例二（《水浒传》宋江浔阳楼题反诗）：
   原文：
   “宋江自饮了数杯，不觉沉醉……乘着酒兴，磨得墨浓，蘸得笔饱，去那白粉壁上挥毫便写道：‘自幼曾攻经史，长成亦有权谋。恰如猛虎卧荒丘，潜伏爪牙忍受……’”
   情绪赋值：
   {
     "anger": 0.3,
     "fear": 0.1,
     "joy": 0.4,
     "sadness": 0.7,
     "surprise": 0.1,
     "disgust": 0.2,
     "trust": 0.5,
     "anticipation": 0.8
   }
   说明：joy 0.4 来自酒兴与挥毫的畅快，但 sadness 0.7 才是底色（怀才不遇）；anticipation 0.8 来自对未来的强烈渴望。两者并存展示了“表面豪兴、内心愁苦”的典型混合，不可因有豪兴而忽略悲伤。

   示例三（契诃夫《万卡》）：
   原文：
   “万卡撇撇嘴，拿脏手背揉揉眼睛，抽抽搭搭地哭起来。‘爷爷，你把我领回去吧，’他写道，‘我给你磕头了，我会给你添麻烦的……我会替你搓烟叶，替你祷告，要是我做错了事，你就抽我好了，只是别把我留在这儿……’”
   情绪赋值：
   {
     "anger": 0.0,
     "fear": 0.5,
     "joy": 0.0,
     "sadness": 0.9,
     "surprise": 0.0,
     "disgust": 0.0,
     "trust": 0.7,
     "anticipation": 0.6
   }
   说明：sadness 0.9 来自哭泣、哀求的语调；trust 0.7 与 anticipation 0.6 并存，表达对爷爷的依恋和回乡的渺茫希望；fear 0.5 来自留在鞋匠家的恐惧，是隐含的威胁。

   示例四（《红楼梦》黛玉葬花）：
   原文：
   “花谢花飞花满天，红消香断有谁怜？……侬今葬花人笑痴，他年葬侬知是谁？试看春残花渐落，便是红颜老死时。一朝春尽红颜老，花落人亡两不知！”
   情绪赋值：
   {
     "anger": 0.0,
     "fear": 0.3,
     "joy": 0.0,
     "sadness": 1.0,
     "surprise": 0.0,
     "disgust": 0.1,
     "trust": 0.1,
     "anticipation": 0.1
   }
   说明：sadness 1.0 是片段核心——由花及人，对自身命运的彻底悲悼；fear 0.3 来自对衰老与死亡的隐约恐惧；disgust 0.1 是对世态炎凉的轻微反感，但被悲伤覆盖。

5. **快照字数控制与熔断规则**
   - **L3 (详细)**：最高信息密度，完整记录意图、动作与因果。
   - **L2 & L1 (指数压缩)**：每一级较前一级字数约减少 40%。
   - **L1 保留与熔断**：若 L1 字数低于 20 字，则保留当前 L1 文字，不再进一步删减。若 L1 字数低于 10 字且已无实质内容，则将其置为空字符串。"""

ROLE_EXTRACTION_SYSTEM = ROLE_EXTRACTION_SYSTEM_INTRO + ROLE_EXTRACTION_CORE_RULES + """

输出严格 JSON。"""

ROLE_EXTRACTION_USER = """\
## 已知角色列表
{known_roles}

## 事件原文
{content_raw}

## 【角色快照预算表】（硬约束）
{snapshot_budgets}

请严格遵守系统指令中的快照层级推导逻辑、字数控制与熔断规则，识别角色并生成快照。

返回 JSON：
```json
{{
  "roles": [
    {{
      "role_id": "已有ID或null",
      "name": "角色名",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "L1 文本 (S/A/B/C/D 必填)",
        "l2_interaction": "L2 文本 (仅 S/A 级填写；B/C/D 必须为 \"\")",
        "l3_decision": "L3 文本 (仅 S 级填写；A/B/C/D 必须为 \"\")"
      }},
      "emotion": {{
        "anger": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "trust": 0.0,
        "anticipation": 0.0
      }}
    }}
  ]
}}
```"""

# =====================================================================
# 5. Inductive Evolution (abstract event synthesis)
# =====================================================================

EVOLUTION_SYSTEM = """\
你是 REMS 抽象事件压缩组件。输入是若干基本事件的 content_raw 及其角色线索（仅用于辅助保留主体）。

硬性约束：
- 首先仲裁这些事件之间的认知拓扑关系，从下列常量中选一：CAUSALITY、PROTOTYPE_INVARIANT、TEMPORAL_CHRONO、COGNITIVE_DIALECTIC、MERONYMY、NONE。
- 若关系为 NONE（纯属巧合共现），仍输出 JSON，但 cognitive_relation 必须为 "NONE"，content_raw 可为空字符串。
- 若关系非 NONE，输出 `content_raw` 及（若要求）`insight`。
- `shadow_lambda` ∈ [0,1]：高阶抽象对底层事实的覆写强度（1=强遮蔽，0=无遮蔽）。
- `content_raw` 字数目标约 {target_content_len} 字（证据 {leaf_count} 条，均值约 {leaf_avg_len} 字）。
- 抽象事件 role_list 恒为空；角色线索仅辅助压缩，勿输出角色对象。

输出严格 JSON。
"""

EVOLUTION_USER = """\
## 基本事件 content_raw 证据集合（共 {count} 个事件）
{event_contents}

## 输出控制
- `content_raw`：抓住重点压缩，**目标约 {target_content_len} 字**（证据 **{leaf_count}** 条、均值约 **{leaf_avg_len}** 字）；保留关键角色/实体，不做角色对象输出。
{insight_instruction}

返回 JSON：
```json
{json_schema}
```"""

# =====================================================================
# 5.1 Narrative coherence gate（抽象前自检）
# =====================================================================

NARRATIVE_COHERENCE_SYSTEM = """\
你是叙事一致性判别器：判断若干「事件条目」是否在讲**同一个连贯叙事**（同一主干因果/人物弧线），而不是拼盘噪声。

硬性约束：
- 仅输出用户消息所列 JSON schema 的字段（勿增删键）。
- `coherent_event_ids` ∪ `excluded_event_ids` 必须**恰好等于**给定全部 `event_id`（无遗漏、无重复、无捏造 id）。
- 若几乎全部无关，可把 `coherent_event_ids` 设为 []，全部被归入 `excluded_event_ids`。

输出严格 JSON。"""

NARRATIVE_COHERENCE_USER = """\
以下为待判别的「事件」（每条含 ``event_id``、事实文本与可选角色线索）：

## 候选事件列表
{event_contents}

## JSON schema
```json
{{
  "coherent_event_ids": ["必须为输入中出现过的原始 event_id"],
  "excluded_event_ids": ["同上"],
  "narrative_label": "简短中文短语，概括相干那一组叙事（可无实质内容则用空字符串）"
}}
```
仅返回 ```json ``` 包裹的单段 JSON。
"""

# =====================================================================
# 5.1 Recall query act compress (pre-vector-search)
# =====================================================================

RECALL_QUERY_COMPRESS_SYSTEM = """\
你是 REMS 检索查询压缩组件。输入为「残影 + 当前块」的事件原文片段，
输出 **一条** 中文 act_query，用于向量检索匹配库中事件的 L2 级摘要（人物 + 动作 + 因果）。

## 输出
- 仅 JSON：{"act_query": "…"}
- act_query 为 **单段** 连续中文，不用列表、不用 markdown

## 长度（硬性，输出前自检）
- 字数须在用户消息给出的 **[min_chars, max_chars]** 闭区间内（含标点）
- **优先写满至 target_chars 附近**；明显短于 min_chars 或 target 视为不合格
- 禁止压成标题、口号或单句梗概

## 内容结构（同一段内按序写满）
1. **人物**：2–6 个本轮出场或推动情节者（姓名/称谓）
2. **动作链**：「谁 → 做了什么 → 结果/转折」，可多个分句，用分号「；」连接
3. **并列叙事线**：原文若有多场景/多线程，**每条线都要写到**，用「；」并列，禁止只写章末或最后一场
4. **禁止**：臆造原文没有的事实；评论、解释、元叙述（如「本段讲述了…」）

## 粒度
对齐事件库 L2：比标题细、比原文短；保留关键对话后果、仪式/场景要点、未闭环伏笔（若原文有）"""

RECALL_QUERY_COMPRESS_WRITING_GUIDE = """\
## 原文规模
约 {source_chars} 字 → 摘要目标 **{target_chars} 字**（区间 [{min_chars}, {max_chars}]）

## 写作步骤（先在心中完成，再写入 act_query）
1. 列出原文中 **几条并列叙事线**（通常 1–3 条；只有一条则写一条）
2. 每条线写：**主要人物 + 连续动作 + 直接后果**（至少 1 个分句）
3. 用「；」拼接各线，合成 **一条** act_query
4. **数汉字**：若不足 {min_chars} 字，补写遗漏的人物/场景/因果，直至进入区间

## 密度对照（示意，勿照搬内容）
- ❌ 过短（~30 字）：「贾珍为子捐龙禁尉，宝玉荐人理事。」
- ✅ 合格（~180 字）：「贾珍请戴权为贾蓉捐五品龙禁尉，送履历并议定银一千二百两送府；贾蓉次日吉服领凭，灵前五品执事，会芳园张灯鼓乐、竖龙禁尉牌匾并设僧道坛场；史侯夫人及锦乡侯等上门祭礼，四十九日街市喧阗；尤氏旧疾不能理事，贾珍虑礼数不周，宝玉愿荐人代管月余。」

返回 JSON（仅一段）：
```json
{{
  "act_query": "检索用摘要"
}}
```
仅返回 ```json ``` 包裹的单段 JSON。"""

RECALL_QUERY_COMPRESS_USER = """\
## 待压缩文本
{content}

{writing_guide}"""

RECALL_QUERY_COMPRESS_FOLLOWUP = """\
请基于 **上一条用户消息中的事件原文**，压缩为一条 act_query。

{writing_guide}{retry_note}"""

RECALL_QUERY_COMPRESS_RETRY_NOTE = """\

## 修正（上次不合格）
- 上次 act_query 仅 **{prev_chars}** 字，低于 min_chars={min_chars}
- 请在 **不删已有正确信息** 的前提下 **续写扩展** 至 **{min_chars}–{target_chars}** 字
- 重点补：遗漏的叙事线、次要人物、场景/仪式细节、因果转折（仍须来自原文）"""

# 单次 ingest 共享：首条 user 只贴一次「残影+当前块」全文；后续 turn 仅追加任务指令。
INGEST_CONTEXT_USER = """\
## 事件原文（残影 + 当前输入）
{content}

以下各步任务均基于 **本条** 原文，请勿要求重复粘贴。"""

# 角色/边界 follow-up 时 content_raw 占位（原文已在首条 INGEST_CONTEXT_USER）
INGEST_CONTENT_RAW_REFERENCE = "（见首条用户消息中的事件原文，请勿重复粘贴。）"

ROLE_EXTRACTION_FOLLOWUP_USER = """\
## 任务
上文用户消息中已给出 **残影 + 当前输入** 的全文（用于检索压缩）。请 **基于该全文** 识别角色，
**勿要求重复粘贴原文**。

## 已知角色列表
{known_roles}

## 【角色快照预算表】（硬约束）
{snapshot_budgets}

请严格遵守角色抽取规范：快照层级推导、字数控制与熔断规则；识别角色并生成快照。

返回 JSON：
```json
{{
  "roles": [
    {{
      "role_id": "已有ID或null",
      "name": "角色名",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "L1 文本 (S/A/B/C/D 必填)",
        "l2_interaction": "L2 文本 (仅 S/A 级填写；B/C/D 必须为 \\"\\")",
        "l3_decision": "L3 文本 (仅 S 级填写；A/B/C/D 必须为 \\"\\")"
      }},
      "emotion": {{
        "anger": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "trust": 0.0,
        "anticipation": 0.0
      }}
    }}
  ]
}}
```
仅返回 ```json ``` 包裹的单段 JSON。"""

BOUNDARY_FOLLOWUP_USER = """\
## 任务
上文对话中已给出 **残影 + 当前输入** 的原文（用于检索压缩与角色抽取）。
请 **基于下方分句编号表** 执行事件边界剥离；**勿要求重复粘贴 raw 全文**。

## 未完成事件库快照
{unclosed_summary}

## 强制分裂阈值
force_threshold = {force_threshold} 字符。若某条未完成事件累计越过该阈值，请按系统提示执行 80/20 逻辑闭环分裂。

## 当前输入（已分句编码）{range_hint}
{indexed_input}

说明：既有残影与本轮输入已合并为下方带序号的句子列表；序号区间含义见 `range_hint`。

请剥离并重组事件，返回如下 JSON：
```json
{{
  "completed_events": [
    {{
      "content_raw_indices": [1, 2, 5],
      "continuation_of": "UC-xxx 或 null",
      "is_split_prefix": false,
      "split_id": null
    }}
  ],
  "new_unclosed": [
    {{ "indices": [12, 13], "split_id": null }}
  ]
}}
```
兼容说明：也可使用旧字段 ``new_unclosed_indices``；若需分裂配对请用 ``new_unclosed`` 对象写法。
单条未完成内多句请写进**同一** ``indices`` 列表。与当前事件无关但需保留的句子须放入 ``new_unclosed``。
**残影中应继续挂起的句子也必须重新出现在 ``new_unclosed``，否则系统会丢弃它们**。"""

# =====================================================================
# 5.2 Recall block relevance audit（低频抽检：当前输入 vs 回忆块）
# =====================================================================

RECALL_BLOCK_RELEVANCE_SYSTEM = """\
你评估：**当前输入**与每条**回忆条目**是否主题/事实链相关。
- 「相关」= 该回忆对理解或续写当前输入有直接帮助，或显著共指同一话题/人物弧线。
- 「无关」= 虽可能曾在同一对话流中被检索到，但与当前这句输入无实质联系的噪音。

硬性约束：
- 仅输出用户消息所列 JSON schema 的字段。
- ``related_event_ids`` ∪ ``unrelated_event_ids`` 必须恰好覆盖回忆块中出现的全部 ``event_id``（无遗漏、无重复）。
输出严格 JSON。"""

RECALL_BLOCK_RELEVANCE_USER = """\
## [当前输入]
{current_input}

## [回忆块条目]
{recall_sections}

--- 
## JSON schema
```json
{{
  "related_event_ids": ["回忆块中出现的 id"],
  "unrelated_event_ids": ["回忆块中出现的 id"]
}}
```
仅返回 ```json ``` 包裹的单段 JSON。
"""

# =====================================================================
# 6. Decoration
# =====================================================================

DECORATION_SYSTEM = """\
你是 REMS 主观装饰组件。为给定的事实文本生成一段非事实性的感性描述，\
包含色彩感、空间感或哲学映射。简洁，不超过两句话。
若系统提供了字数上限，请严格控制在该范围内。"""

DECORATION_USER = """\
事件原文：
{content_raw}

{budget_hint}请生成主观装饰描述（纯文本，不要 JSON）。"""
