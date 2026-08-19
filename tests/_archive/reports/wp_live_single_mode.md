# WP-LIVE · 单人模式全景叙述
- **报告文件**：`wp_live_single_mode.md`

## 配置快照

```json
{
  "context_window": 4096,
  "chars_per_token": 1.0,
  "len_msg": 62,
  "physical_redline": 620,
  "safe_watermark": 124,
  "user_mode": "single",
  "core_user_role_id": "ROL-wp-live-core",
  "active_participants": [],
  "recall_cluster_threshold": 5,
  "summary_fuse_min_chars": 20,
  "ae_high_threshold": 0.6,
  "ae_score_weight": 0.15,
  "activation_energy_gain": 1.0,
  "activation_energy_weight": 0.1,
  "ema_smoothing_alpha": 0.4,
  "wp_primary_field": "l3_decision",
  "wp_default_field": "l2_interaction",
  "wp_half_life_days": 60.0,
  "ae_forgetting_multiplier": 5.0,
  "semantic_card_max_keys": 20,
  "hallucination_anchor_prob": 0.3
}
```

## LLM / 嵌入环境

- **Chroma 嵌入**: `BAAI/bge-small-zh-v1.5`（`torch` 2.4.1 · 解释器: `C:\Users\40575\anaconda3\envs\py3125\python.exe`）
- **剧本截断**: `REMS_WP_LIVE_NARRATIVE_ROUNDS=5` → 本轮 **5** 条

## 指标核对表

| 指标 | 状态 | 实测摘要 | 章节 |
| --- | --- | --- | --- |
| §1.1.1 event_id 字典序递增 | **PASS** | {"count": 4, "first": ["EVT-0019da0df2728-1d2be413", "EVT-0019da0df3dd8-983aba25", "EVT-0019da0df5286-0cddc911"], "last": ["EVT-0019da0df3dd8-983aba25", "EVT-0019da0df5286-0cddc911", "EVT-0019da0df6d9 | §1.1.1 |
| §1.1.3 摘要熔断 | **PASS** | {"actual_max_level": 1, "summary_lengths": {"L1": 14}} | §1.1.3 |
| §1.1.7 event_length ≤ len_msg | **PASS** | 19 | §1.1.7 |
| §1.1.7 防碎片化聚合 | **PASS** | {"hits": ["跑", "打扫", "小说"], "content_preview": "我去跑了步，然后回家，接着打扫了卫生，又看了一小时小说。"} | §1.1.7 |
| §2.2 single 模式提示词注入 | **PASS** | {"单人隔离模式": true, "核心用户": true, "ROL-wp-live-core": true} | §2.2 |
| §2.3 动态遗忘：高 AE > 低 AE 有效留存 | **PASS** | {"high_avg": 0.8998, "low_avg": 0.5043, "high_count": 1, "low_count": 1} | §2.3 |
| §2.4 语义卡片键上限 | **PASS** | 0 | §2.4 |
| §2.5 EMA | **INFO** | 无捕获（首轮或无历史） | §2.5 |
| §2.5 activation_energy 硬绑定 | **PASS** | {"ae": 1.0, "activation_energy": 1.0} | §2.5 |
| §2.5 activation_energy 硬绑定 | **PASS** | {"ae": 1.0, "activation_energy": 1.0} | §2.5 |
| §4.4 回忆块 ≤ physical_redline | **PASS** | 48 | §4.4 |
| §4.4 ultra fallback 触发 | **FAIL** | {"levels": ["L1", "L1", "L1", "L1"], "has_ultra": false} | §4.4 |

## §1.1.1 event_id 递增

### 白皮书原文

> 生成字典序大致随时间递增、全局唯一的 `event_id`（`EVT-` 前缀）。
### 检查方法

- 按封存顺序收集基本事件 ID，检查非递减排序。
### 现场数据 · §1.1.1 event_id 字典序递增 **`PASS`**

- **期望**：封存顺序下 event_id 列表整体呈升序
- **实测**：`{"count": 4, "first": ["EVT-0019da0df2728-1d2be413", "EVT-0019da0df3dd8-983aba25", "EVT-0019da0df5286-0cddc911"], "last": ["EVT-0019da0df3dd8-983aba25", "EVT-0019da0df5286-0cddc911", "EVT-0019da0df6d9`

## §1.1.3 递归摘要熔断

### 白皮书原文

> 递归摘要应在字数不足 `summary_fuse_min_chars` 时熔断。
### 检查方法

- 读取最后一条封存事件的 `actual_max_level` 与 `summary_lengths`。
### 现场数据 · §1.1.3 摘要熔断 **`PASS`**

- **期望**：末级摘要字数 < 20 或 actual_max_level == 5
- **实测**：`{"actual_max_level": 1, "summary_lengths": {"L1": 14}}`


```
{
  "event_id": "EVT-0019da0df6d95-a901c6e6",
  "tail_len": 14,
  "beyond_tail": false
}
```

## §1.1.7 event_length 上限

### 白皮书原文

> 单事件 `event_length` 不得超过 `len_msg`。
### 现场数据 · §1.1.7 event_length ≤ len_msg **`PASS`**

- **期望**：≤ 62
- **实测**：`19`


```
{
  "event_id": "EVT-0019da0df6d95-a901c6e6"
}
```

## §1.1.7 防碎片化聚合

### 白皮书原文

> 对于描述同一连续时间段内、性质相近的琐碎日常动作……不得将其拆分为多个极短的独立事件。
### 检查方法

- 使用含跑步→回家→打扫→看小说的单轮输入诱导合并。
### 现场数据 · §1.1.7 防碎片化聚合 **`PASS`**

- **期望**：同一事件的 content_raw 同时包含 ≥2 枚关键字：['跑', '打扫', '小说']
- **实测**：`{"hits": ["跑", "打扫", "小说"], "content_preview": "我去跑了步，然后回家，接着打扫了卫生，又看了一小时小说。"}`


```
{
  "event_id": "EVT-0019da0df3dd8-983aba25"
}
```
### 合并事件摘录

```json
{
  "event_id": "EVT-0019da0df3dd8-983aba25",
  "l0": "我去跑了步，然后回家，接着打扫了卫生，又看了一小时小说。"
}
```

## §2.2 单人模式提示词注入

### 白皮书原文

> 单人隔离模式下，system prompt 应注入核心用户与模式说明（白皮书 2.2）。
### 检查方法

- 拦截首次 `role_extraction` 的 system 消息。
### 现场数据 · §2.2 single 模式提示词注入 **`PASS`**

- **期望**：system prompt 包含：['单人隔离模式', '核心用户', 'ROL-wp-live-core']
- **实测**：`{"单人隔离模式": true, "核心用户": true, "ROL-wp-live-core": true}`


```
{
  "system_prompt_head": "【全局运行模式：单人隔离模式（Single-User）】\n当前系统运行于单人隔离模式。上下文中所有第一人称代词（\"我\"、\"我的\"）以及缺乏明确主语的动作，\n均极大可能指向唯一核心用户实体 <ROL-wp-live-core>。\n- 在抽取角色/切分边界时，应优先将此类模糊主语归并为该核心用户；\n- 除非文本中出现明确的第三方人名/称谓，否则不应新增其他角色。\n\n你是 REMS 角色提取组件。从事件原文中识别所有参与实体（人物、动物、关键物体），并为每个角色生成快照和情感量化。\n\n要求：\n1. 替换所有模糊代词（他/她/它/他们等）为明确角色名。\n2. 对每个角色评定重要性：S/A/B/C/D。\n3. 生成三级快照：\n   - L1（提及）：是否出现或被提及。\n   - L2（互动）：核心行为、言论。\n   - L3（决策）：决策逻辑、行为倾向。\n4. 量化当次事件中该角色的情感状态。\n\n输出严格 JSON。"
}
```
### system prompt 前 300 字

```json
"【全局运行模式：单人隔离模式（Single-User）】\n当前系统运行于单人隔离模式。上下文中所有第一人称代词（\"我\"、\"我的\"）以及缺乏明确主语的动作，\n均极大可能指向唯一核心用户实体 <ROL-wp-live-core>。\n- 在抽取角色/切分边界时，应优先将此类模糊主语归并为该核心用户；\n- 除非文本中出现明确的第三方人名/称谓，否则不应新增其他角色。\n\n你是 REMS 角色提取组件。从事件原文中识别所有参与实体（人物、动物、关键物体），并为每个角色生成快照和情感量化。\n\n要求：\n1. 替换所有模糊代词（他/她/它/他们等）为明确角色名。\n2. 对每个角色评定重要性：S/A/B/C/D。"
```

## §2.3 白描收集粒度

### 白皮书原文

> 主角取 L3 决策白描，配角取 L2 互动白描。
### 检查方法

- 抽样含多角色的事件，对照 `WhitePaintingEntry` 与 `role_snapshot`。
### 备注

_未找到多角色白描对齐样本，可能模型未拆分角色。_

## §2.3 动态遗忘（半衰）

### 白皮书原文

> 高 AE 条目半衰期按 `ae_forgetting_multiplier` 拉长。
### 现场数据 · §2.3 动态遗忘：高 AE > 低 AE 有效留存 **`PASS`**

- **期望**：avg(high_ae_retention) > avg(low_ae_retention)
- **实测**：`{"high_avg": 0.8998, "low_avg": 0.5043, "high_count": 1, "low_count": 1}`

## §2.4 语义卡片键上限

### 白皮书原文

> 语义卡片键数不得超过 `semantic_card_max_keys`。
### 现场数据 · §2.4 语义卡片键上限 **`PASS`**

- **期望**：≤ 20
- **实测**：`0`


```
{
  "keys": []
}
```
### 最近卡片 data 快照

```json
[]
```

## §2.5 EMA 指数平滑

### 白皮书原文

> smoothed = α·current + (1-α)·history（白皮书 2.5）。
### 检查方法

- 对核心用户记录历史均值 joy、当前 joy、平滑后 joy。
### 现场数据 · §2.5 EMA **`INFO`**（无硬判定）

- **期望**：至少一轮存在白描历史后的封存
- **实测**：`无捕获（首轮或无历史）`

## §2.5 activation_energy 硬绑定

### 白皮书原文

> 高 AE 事件的 `activation_energy` 应体现重大事件硬绑定（×1.5 增益上限）。
### 检查方法

- 筛选 `affective_energy ≥ ae_high_threshold` 的封存事件。
### 现场数据 · §2.5 activation_energy 硬绑定 **`PASS`**

- **期望**：≥ min(1, ae·gain·1.5) = 1.000
- **实测**：`{"ae": 1.0, "activation_energy": 1.0}`


```
{
  "event_id": "EVT-0019da0df5286-0cddc911",
  "gain": 1.0
}
```
### 现场数据 · §2.5 activation_energy 硬绑定 **`PASS`**

- **期望**：≥ min(1, ae·gain·1.5) = 1.000
- **实测**：`{"ae": 1.0, "activation_energy": 1.0}`


```
{
  "event_id": "EVT-0019da0df6d95-a901c6e6",
  "gain": 1.0
}
```

## §4.4 回忆块预算与混合分

### 白皮书原文

> 回忆块总长应受 `physical_redline` 约束；混合分为余弦/时间/角色/AE/Act 加权。
### 检查方法

- 最后一轮 `ContextPackage.recall_block` 与 `hybrid_scores` 轨迹。
### 现场数据 · §4.4 回忆块 ≤ physical_redline **`PASS`**

- **期望**：≤ 620 字
- **实测**：`48`


```
{
  "items": 4,
  "levels": [
    "L1",
    "L1",
    "L1",
    "L1"
  ]
}
```
**Top 混合分分量（近期轨迹）**

```text
`EVT-0019da0df2728-1d2be413` cosine:0.193 | time:0.150 | role:0.060 | ae:0.075 | act:0.050 → total=0.528
`EVT-0019da0df2728-1d2be413` cosine:0.200 | time:0.150 | role:0.060 | ae:0.075 | act:0.050 → total=0.535
`EVT-0019da0df3dd8-983aba25` cosine:0.151 | time:0.150 | role:0.060 | ae:0.000 | act:0.000 → total=0.361
`EVT-0019da0df2728-1d2be413` cosine:0.200 | time:0.150 | role:0.060 | ae:0.075 | act:0.050 → total=0.535
`EVT-0019da0df3dd8-983aba25` cosine:0.177 | time:0.150 | role:0.060 | ae:0.000 | act:0.000 → total=0.387
`EVT-0019da0df5286-0cddc911` cosine:0.162 | time:0.150 | role:0.060 | ae:0.150 | act:0.100 → total=0.622
`EVT-0019da0df5286-0cddc911` cosine:0.218 | time:0.150 | role:0.060 | ae:0.150 | act:0.100 → total=0.678
`EVT-0019da0df6d95-a901c6e6` cosine:0.211 | time:0.150 | role:0.060 | ae:0.150 | act:0.100 → total=0.671
`EVT-0019da0df2728-1d2be413` cosine:0.211 | time:0.150 | role:0.060 | ae:0.075 | act:0.050 → total=0.546
`EVT-0019da0df3dd8-983aba25` cosine:0.167 | time:0.150 | role:0.060 | ae:0.000 | act:0.000 → total=0.377
```
**RecallBlock Top-K**

- `EVT-0019da0df5286-0cddc911` score=0.6776 level=L1
- `EVT-0019da0df6d95-a901c6e6` score=0.6706 level=L1
- `EVT-0019da0df2728-1d2be413` score=0.5455 level=L1
- `EVT-0019da0df3dd8-983aba25` score=0.3766 level=L1

## §4.4 ultra 降级

### 白皮书原文

> 当总字数逼近上限时，启用 ultra 极简映射。
### 检查方法

- 使用窄 `physical_redline` 的临时配置强行组装回忆块。
### 现场数据 · §4.4 ultra fallback 触发 **`FAIL`**

- **期望**：items 中至少一条 summary_level == 'ultra'
- **实测**：`{"levels": ["L1", "L1", "L1", "L1"], "has_ultra": false}`

## 运行摘要

- 剧本轮次：**5**（全量剧本共 18 条）
- 新封存事件数：**4**
- LLM 调用次数：**22**

## LLM 调用记录

数据来自 **`LLMProvider.invocation_history()`**（每次 ``complete`` 一行）。

**延迟**为客户端测量的往返时间（毫秒）；**token** 取自 API `usage`，网关未返回时显示为「—」。

| # | task_type | model | 延迟(ms) | prompt | completion | total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `boundary_detection` | `qwen-turbo` | 1034.82 | 571 | 55 | 626 |
| 2 | `summary` | `tongyi-xiaomi-analysis-pro` | 733.62 | 201 | 29 | 230 |
| 3 | `role_extraction` | `qwen-turbo` | 2396.39 | 493 | 188 | 681 |
| 4 | `summary` | `tongyi-xiaomi-analysis-pro` | 568.30 | 78 | 22 | 100 |
| 5 | `summary` | `tongyi-xiaomi-analysis-pro` | 588.00 | 166 | 21 | 187 |
| 6 | `boundary_detection` | `qwen-turbo` | 846.66 | 577 | 61 | 638 |
| 7 | `summary` | `tongyi-xiaomi-analysis-pro` | 733.08 | 207 | 29 | 236 |
| 8 | `role_extraction` | `qwen-turbo` | 2487.74 | 499 | 217 | 716 |
| 9 | `summary` | `tongyi-xiaomi-analysis-pro` | 942.41 | 84 | 39 | 123 |
| 10 | `summary` | `tongyi-xiaomi-analysis-pro` | 809.53 | 215 | 35 | 250 |
| 11 | `boundary_detection` | `qwen-turbo` | 849.33 | 568 | 52 | 620 |
| 12 | `summary` | `tongyi-xiaomi-analysis-pro` | 623.48 | 198 | 24 | 222 |
| 13 | `role_extraction` | `qwen-turbo` | 2158.41 | 490 | 201 | 691 |
| 14 | `summary` | `tongyi-xiaomi-analysis-pro` | 735.82 | 75 | 32 | 107 |
| 15 | `summary` | `tongyi-xiaomi-analysis-pro` | 904.33 | 266 | 43 | 309 |
| 16 | `boundary_detection` | `qwen-turbo` | 844.01 | 575 | 59 | 634 |
| 17 | `summary` | `tongyi-xiaomi-analysis-pro` | 717.93 | 205 | 31 | 236 |
| 18 | `role_extraction` | `qwen-turbo` | 3658.62 | 497 | 355 | 852 |
| 19 | `summary` | `tongyi-xiaomi-analysis-pro` | 677.08 | 82 | 26 | 108 |
| 20 | `summary` | `tongyi-xiaomi-analysis-pro` | 1467.70 | 303 | 54 | 357 |
| 21 | `summary` | `tongyi-xiaomi-analysis-pro` | 1235.31 | 343 | 65 | 408 |
| 22 | `boundary_detection` | `qwen-turbo` | 879.89 | 568 | 63 | 631 |

**Token 合计（仅统计 API 返回了对应字段的调用）**：prompt Σ=7261 · completion Σ=1701 · total Σ=8962

## 环境 & 复现命令

- **生成时间**：2026-04-18 13:54:27 UTC
- **Python**：`C:\Users\40575\anaconda3\envs\py3125\python.exe`
- **pytest 命令**：`python -m pytest tests/test_wp_live_single_mode_narrative.py -v -s  # REMS_WP_LIVE_NARRATIVE_ROUNDS=5`
- **相关环境变量键**（值不落盘）：`REMS_LIVE_ALLOW_FAKE_EMBEDDING`、`REMS_RUN_LIVE_METABOLISM_TEST`、`REMS_WP_REPORT_DIR`

<!-- wp-live-stats: {"report": "wp_live_single_mode.md", "pass": 10, "fail": 1, "na": 1} -->
