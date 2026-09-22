# cg-201 提示词对照草稿（原 REMS3 vs 本仓）

> 目录：`design/refactor/`｜状态：**草稿**
>
> 原版：`C:/Users/40575/Desktop/项目/memory/rems_3/src/rems/llm/prompts.py`
> 本仓：`c:/Users/40575/Desktop/prog/Jshi_memory/src/rems/llm/prompts.py`
>
> 范围：封装主路径相关（边界切分 / 超长分裂 / 事件充实）。
> 说明：下列为**模板常量正文**；实际下发时还会前置 `build_user_mode_block`，
> 并对 `{...}` 占位符做 `.format`（见 `_tmp_seal_out/llm_calls/00_assembly_rules.md`）。

## 0. 索引与差异摘要

| 常量 | 原版 | 本仓 | 关系 |
|------|------|------|------|
| `BOUNDARY_SYSTEM` | 有 | 有 | 相同 |
| `BOUNDARY_USER` | 有 | 有 | 相同 |
| `BOUNDARY_FOLLOWUP_USER` | 有 | 有 | 相同 |
| `OVERLONG_UC_SPLIT_SYSTEM` | 有 | 有 | 相同 |
| `OVERLONG_UC_SPLIT_USER` | 有 | 有 | 相同 |
| `ENRICHMENT_SUMMARY_A_RULES` | 有 | 有 | **有差异** |
| `ENRICHMENT_MEMORY_INTRO` | 无 | 有 | 仅本仓（匠石 memory 路径） |
| `ENRICHMENT_MEMORY_OBJECT_RULES` | 无 | 有 | 仅本仓（匠石 memory 路径） |
| `ENRICHMENT_MEMORY_EMOTION_EXAMPLES` | 无 | 有 | 仅本仓（匠石 memory 路径） |
| `ENRICHMENT_MEMORY_USER` | 无 | 有 | 仅本仓（匠石 memory 路径） |
| `ENRICHMENT_SUMMARY_ONLY_INTRO` | 有 | 有 | 相同 |
| `ENRICHMENT_SUMMARY_ONLY_SYSTEM_SUFFIX` | 有 | 有 | 相同 |
| `ENRICHMENT_SUMMARY_ONLY_USER` | 有 | 有 | 相同 |
| `ENRICHMENT_FULL_ROLE_BRIDGE` | 有 | 有 | 相同 |
| `ENRICHMENT_FULL_ROLE_FOOTER` | 有 | 有 | 相同 |
| `ENRICHMENT_FULL_USER` | 有 | 有 | 相同 |

### 拼接提醒（本仓写入）

```text
# 边界
system = build_user_mode_block(config) + BOUNDARY_SYSTEM
user   = BOUNDARY_USER.format(unclosed_summary, force_threshold, range_hint, indexed_input)

# 充实（objects 非 None → memory）
system = build_user_mode_block(config)
       + ENRICHMENT_MEMORY_INTRO
       + ENRICHMENT_SUMMARY_A_RULES.format(fuse_*)
       + ENRICHMENT_MEMORY_OBJECT_RULES
       + ENRICHMENT_MEMORY_EMOTION_EXAMPLES
user   = ENRICHMENT_MEMORY_USER.format(content_raw, object_list, summary_budget_table)
```

---

## `BOUNDARY_SYSTEM`

*原版与本仓相同。*

```
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

输出严格 JSON。
```

---

## `BOUNDARY_USER`

*原版与本仓相同。*

```
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
单条未完成也可写 ``{{"indices": [12]}}``（同一条内多句请写进**同一**扁平列表，勿每句一条记录）。与当前事件无关但需保留的句子，也请放入 ``new_unclosed``。**残影中应继续挂起的句子也必须重新出现在 ``new_unclosed``，否则系统会丢弃它们**。
```

---

## `BOUNDARY_FOLLOWUP_USER`

*原版与本仓相同。*

```
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
**残影中应继续挂起的句子也必须重新出现在 ``new_unclosed``，否则系统会丢弃它们**。
```

---

## `OVERLONG_UC_SPLIT_SYSTEM`

*原版与本仓相同。*

```
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
```

---

## `OVERLONG_UC_SPLIT_USER`

*原版与本仓相同。*

```
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
如果实在无法在满足约束下切分，请只返回 ``{{"abort": true}}``。
```

---

## `ENRICHMENT_SUMMARY_A_RULES`

### 原版（rems_3）

```
A. **summaries：L1…Ln 递归压缩**
   - 遵守【摘要字数预算表】；L1 保真主干，必须是一段**通顺的完整叙事**，包含关键动作、因果转折与重要心理细节，不得写成事件清单。
   - L2+ 逐层约减半；每一级应是对**上一级摘要**的语义压缩，而非对原文的重新概括。
   - **熔断规则**：满足任一条件即停止生成下一级，`summaries` 仅含已产出层级：（1）下一级 Ln 的**预算字数** ≤ {fuse_min_chars} 字；（2）上一级 L(n-1) 摘要的**实际字数** × 0.5 < {fuse_compact_threshold} 字（阈值 = floor({fuse_min_chars} × 0.7)，至少 1）。
   - **可读性底线**：任何一级摘要须为语法通顺的完整句子；若在该级预算内无法维持可读性，宁可不生成该级。
```

### 本仓（Jshi_memory）

```
A. **summaries：L1…Ln 递归压缩**
   - 遵守【摘要字数预算表】；L1 保真主干，必须是一段**通顺的完整叙事**，包含关键动作、因果转折与重要心理细节，不得写成事件清单。
   - **可检索性**：L1 必须保留原文的**关键短语、专有名词、数字与术语**（如"两件独立的事""边界调用由 6 次降至 1 次""G 字头 7:05"），供后续回忆检索命中；不得用泛指词替换掉原文的独特指称。
   - L2+ 逐层约减半；每一级应是对**上一级摘要**的语义压缩，而非对原文的重新概括。
   - **熔断规则**：满足任一条件即停止生成下一级，`summaries` 仅含已产出层级：（1）下一级 Ln 的**预算字数** ≤ {fuse_min_chars} 字；（2）上一级 L(n-1) 摘要的**实际字数** × 0.5 < {fuse_compact_threshold} 字（阈值 = floor({fuse_min_chars} × 0.7)，至少 1）。
   - **可读性底线**：任何一级摘要须为语法通顺的完整句子；若在该级预算内无法维持可读性，宁可不生成该级。
```

---

## `ENRICHMENT_MEMORY_INTRO`

*原版无此常量（本仓为匠石契约新增）。*

### 本仓

```
你是匠石记忆后端的事件充实（Event Enrichment）组件。给定**已闭环**经历原文，**单次输出**：
摘要层级（A）、主体（匠石/"我"）事件级情感（B）、对象分级白描（C，匠石视角）。**不抽取角色**——对象由输入映射表给定。

**视角统一**：本记忆是匠石的个人记忆，摘要与对象白描一律以匠石（"我"）第一人称视角叙述，不要混用第三人称"匠石"。
```

---

## `ENRICHMENT_MEMORY_OBJECT_RULES`

*原版无此常量（本仓为匠石契约新增）。*

### 本仓

```
B. **emotion（事件级主体情感）**：给出匠石（主体"我"）在本事件中的 8 维情绪（各 0-1）。情感只属于主体，对象无情感。
C. **objects（匠石对对象的分级白描）**：视角一律是匠石（"我"），三级**各有侧重、不得互相复述**：
   - `l1_mention`：**一句话提及**——事件中该对象在我眼中的状态 / 表现（最简，不展开过程；避免"在场"这类空间化措辞）；
   - `l2_interaction`：**互动过程**——我（匠石）与该对象在本事件中的具体往来：谁先发起、我说了什么 / 对方回应了什么、做了什么（无互动写"（无互动）"）；
   - `l3_decision`：**意图 / 决策及原因**——我眼中该对象在本事件中的选择与动机，必须从 L2 的互动**推导**，且理由须**原文可支撑**；原文无法支撑的动机**不要编造**，写空字符串（而非 L1 / L2 的改写）。
   对象不带情感、不带等级；不要新增名单之外的对象，不要遗漏名单中的对象。
D. **location（可选）**：事件发生地点；无法判断时为 null。

输出严格 JSON。
```

---

## `ENRICHMENT_MEMORY_EMOTION_EXAMPLES`

*原版无此常量（本仓为匠石契约新增）。*

### 本仓

```
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
说明：joy 0.8 来自成果达成；trust 0.6 来自对系统与自身积累的信心；anticipation 0.7 来自对后续的期待；surprise 0.2 来自结果略超预期；sadness 0.1 / fear 0.1 为底色。
```

---

## `ENRICHMENT_MEMORY_USER`

*原版无此常量（本仓为匠石契约新增）。*

### 本仓

```
## 事件原文
{content_raw}

## 对象名单（来自输入映射表；每个都要给一句话快照）
{object_list}

## 摘要字预算
{summary_budget_table}

## 任务
1. 生成 `summaries`（L1 起；递归压缩与熔断规则见系统提示 **A**）。
2. 生成 `emotion`（主体 8 维情绪）。
3. 生成 `objects`（每个对象的分级白描 l1/l2/l3，匠石视角）。
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
    {{"name": "对象名", "l1_mention": "…", "l2_interaction": "…", "l3_decision": "…"}}
  ],
  "location": "地点或 null"
}}
```
```

---

## `ENRICHMENT_SUMMARY_ONLY_INTRO`

*原版与本仓相同。*

```
你是 REMS 事件充实（Event Enrichment）组件。给定**已闭环**事件原文（基本事件或抽象事件的压缩主干均可），本次**仅**输出摘要层级：
```

---

## `ENRICHMENT_SUMMARY_ONLY_SYSTEM_SUFFIX`

*原版与本仓相同。*

```
勿输出 `roles`、`role_list` 或其它字段。只输出一个 JSON 对象，包含 `summaries` 字典。输出严格 JSON。
```

---

## `ENRICHMENT_SUMMARY_ONLY_USER`

*原版与本仓相同。*

```
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
```
```

---

## `ENRICHMENT_FULL_ROLE_BRIDGE`

*原版与本仓相同。*

```
B. **roles：** 下列内容与 REMS「角色提取」（第一步 ``RoleExtractionSkill``）的系统提示 **完全一致**（同一段 ``ROLE_EXTRACTION_CORE_RULES``）：
```

---

## `ENRICHMENT_FULL_ROLE_FOOTER`

*原版与本仓相同。*

```

单次响应须在同一个 JSON 对象中同时给出 `summaries`（递归摘要字典）与 `roles`（角色数组）。顶层形状与用户消息中的 JSON 示例一致。输出严格 JSON。
```

---

## `ENRICHMENT_FULL_USER`

*原版与本仓相同。*

```
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
```

---

## 附：全局模式头（两边同构）

实际 `system` 还会在最前面拼 `build_user_mode_block(config)`。
默认 `user_mode=single` 时注入「单人隔离模式」段；`multi` 时注入花名册段。
正文见 `prompts.py` 中 `_SINGLE_MODE_TEMPLATE` / `_MULTI_MODE_TEMPLATE`。
