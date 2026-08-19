# WP-LIVE · 多人与多场景输出
- **报告文件**：`wp_live_multi_and_modes.md`

## 配置快照

```json
{
  "context_window": 32768,
  "chars_per_token": 1.5,
  "len_msg": 744,
  "physical_redline": 7447,
  "safe_watermark": 1489,
  "user_mode": "multi",
  "core_user_role_id": null,
  "active_participants": [
    "Alice",
    "Bob",
    "Carol"
  ],
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
- **多人剧本**: 本轮 **5** 条（全量 5）

## 指标核对表

| 指标 | 状态 | 实测摘要 | 章节 |
| --- | --- | --- | --- |
| §2.2 multi 模式提示词注入 | **PASS** | {"多人交互模式": true, "活跃参与者": true, "Alice": true, "Bob": true} | §2.2 |
| §2.2 抽取角色数 ≥2 | **PASS** | 3 | §2.2 |
| §5.1 DIALOGUE context_package 非空 | **PASS** | {"is_none": false} | §5.1 |
| §4.4 回忆块 ≤ physical_redline | **PASS** | 12 | §5.1 |
| §5.2 PASSIVE_LOG 静默 | **PASS** | {"is_none": true, "sealed": 6} | §5.2 |
| §5.3 NPC_AGENT 结构化指令 | **PASS** | [{"npc_role_id": "ROL-npc-guard", "action": "flee", "klesha_delta": {"anger": 0.3, "ignorance": -0.1}, "reason": "threat_score=0.60"}] | §5.3 |
| §5.3 NPC action ∈ watch/flee | **PASS** | flee | §5.3 |

## §2.2 多人模式提示词注入

### 白皮书原文

> 多人交互模式应注入活跃参与者花名册，禁止默认归一核心用户。
### 检查方法

- 拦截 `role_extraction` system prompt，校验模板关键字。
### 现场数据 · §2.2 multi 模式提示词注入 **`PASS`**

- **期望**：system prompt 包含：['多人交互模式', '活跃参与者', 'Alice', 'Bob']
- **实测**：`{"多人交互模式": true, "活跃参与者": true, "Alice": true, "Bob": true}`


```
{
  "system_prompt_head": "【全局运行模式：多人交互模式（Multi-User）】\n当前为多实体交互场景，已知活跃参与者包含 <Alice、Bob、Carol>。\n请务必结合对话历史、时间戳与发言者账号/声纹标签，执行精确的多方代词消解\n（Multiparty Coreference Resolution）。严禁将模糊代词默认归属于单一核心用户；\n若无法确定代词指代，应如实在输出中标注未决并保留原词，不得臆测。\n\n你是 REMS 角色提取组件。从事件原文中识别所有参与实体（人物、动物、关键物体），并为每个角色生成快照和情感量化。\n\n要求：\n1. 替换所有模糊代词（他/她/它/他们等）为明确角色名。\n2. 对每个角色评定重要性：S/A/B/C/D。\n3. 生成三级快照：\n   - L1（提及）：是否出现或被提及。\n   - L2（互动）：核心行为、言论。\n   - L3（决策）：决策逻辑、行为倾向。\n4. 量化当次事件中该角色的情感状态。\n\n输出严格 JSON。"
}
```
### system prompt 摘要

```json
"【全局运行模式：多人交互模式（Multi-User）】\n当前为多实体交互场景，已知活跃参与者包含 <Alice、Bob、Carol>。\n请务必结合对话历史、时间戳与发言者账号/声纹标签，执行精确的多方代词消解\n（Multiparty Coreference Resolution）。严禁将模糊代词默认归属于单一核心用户；\n若无法确定代词指代，应如实在输出中标注未决并保留原词，不得臆测。\n\n你是 REMS 角色提取组件。从事件原文中识别所有参与实体（人物、动物、关键物体），并为每个角色生成快照和情感量化。\n\n要求：\n1. 替换所有模糊代词（他/她/它/他们等）为明确角色名。\n2. 对每个角色评定重要性：S/A/B/C/D。\n3. 生成三级快照：\n   - L1（提及）：是否出现或被提及。\n   - L2（互动）：核心行为、言论。\n   - L3（决策）：决策逻辑、行为倾向。\n4. 量化当次事件中该角色的情感状态。\n\n输出严格 JSON。"
```

## §2.2 多人抽取规模（陈列）

### 白皮书原文

> 模型对多方代词的实际消解效力需人工审阅；此处仅统计抽取角色数。
### 检查方法

- 读取最近一条封存事件的 `role_list` 长度。
### 现场数据 · §2.2 抽取角色数 ≥2 **`PASS`**

- **期望**：同一输入下 roles ≥2
- **实测**：`3`


```
[
  "null",
  "null",
  "null"
]
```

## §2.3 多人配角白描粒度

### 白皮书原文

> 多人模式无核心用户时，配角白描应走 `l2_interaction` 档位（非 S/A）。
### 检查方法

- 对非 S/A 角色的白描条目执行 `white_painting_tier_check(..., is_primary=False)`。
### 备注

_未找到合适的非 S/A 白描样本。_

## §5.1 DIALOGUE 上下文包

### 白皮书原文

> 标准交互模式应组装 Context Package 供上游生成回复。
### 检查方法

- `context_package is not None`，且 `recall_block.total_length ≤ physical_redline`。
### 现场数据 · §5.1 DIALOGUE context_package 非空 **`PASS`**

- **期望**：context_package is not None
- **实测**：`{"is_none": false}`
### 现场数据 · §4.4 回忆块 ≤ physical_redline **`PASS`**

- **期望**：≤ 7447 字
- **实测**：`12`


```
{
  "items": 1,
  "levels": [
    "L1"
  ]
}
```

## §5.2 PASSIVE_LOG 静默

### 白皮书原文

> 被动日志模式只记不说，不组装 Context Package。
### 检查方法

- `context_package is None`；仍尝试 `force_save` 推动封存。
### 现场数据 · §5.2 PASSIVE_LOG 静默 **`PASS`**

- **期望**：context_package is None
- **实测**：`{"is_none": true, "sealed": 6}`

## §5.3 NPC_AGENT 指令

### 白皮书原文

> NPC 模式输出结构化 `action` 与 `klesha_delta`（白皮书 5.3）。
### 检查方法

- 预置多条回忆文本包含 npc_role_id 以抬高威胁启发分。
### 现场数据 · §5.3 NPC_AGENT 结构化指令 **`PASS`**

- **期望**：npc_directives 非空且每条含 action + klesha_delta
- **实测**：`[{"npc_role_id": "ROL-npc-guard", "action": "flee", "klesha_delta": {"anger": 0.3, "ignorance": -0.1}, "reason": "threat_score=0.60"}]`
### 现场数据 · §5.3 NPC action ∈ watch/flee **`PASS`**

- **期望**：action 为 watch 或 flee（威胁阈值示例）
- **实测**：`flee`


```
[
  {
    "npc_role_id": "ROL-npc-guard",
    "action": "flee",
    "klesha_delta": {
      "anger": 0.3,
      "ignorance": -0.1
    },
    "reason": "threat_score=0.60"
  }
]
```
### npc_directives

```json
[
  {
    "npc_role_id": "ROL-npc-guard",
    "action": "flee",
    "klesha_delta": {
      "anger": 0.3,
      "ignorance": -0.1
    },
    "reason": "threat_score=0.60"
  }
]
```

## 运行摘要

- 多人 DIALOGUE 剧本：**5** 轮
- 剧本结束后基本事件数：**2**
- 全程 LLM 调用：**83**
- NPC 指令条数：**1**

## LLM 调用记录

数据来自 **`LLMProvider.invocation_history()`**（每次 ``complete`` 一行）。

**延迟**为客户端测量的往返时间（毫秒）；**token** 取自 API `usage`，网关未返回时显示为「—」。

| # | task_type | model | 延迟(ms) | prompt | completion | total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `boundary_detection` | `qwen-turbo` | 1710.50 | 582 | 117 | 699 |
| 2 | `summary` | `tongyi-xiaomi-analysis-pro` | 688.89 | 200 | 28 | 228 |
| 3 | `role_extraction` | `qwen-turbo` | 6363.35 | 486 | 629 | 1115 |
| 4 | `summary` | `tongyi-xiaomi-analysis-pro` | 824.12 | 77 | 37 | 114 |
| 5 | `summary` | `tongyi-xiaomi-analysis-pro` | 286.27 | 165 | 1 | 166 |
| 6 | `summary` | `tongyi-xiaomi-analysis-pro` | 277.43 | 190 | 1 | 191 |
| 7 | `summary` | `tongyi-xiaomi-analysis-pro` | 707.03 | 215 | 29 | 244 |
| 8 | `summary` | `tongyi-xiaomi-analysis-pro` | 738.71 | 264 | 30 | 294 |
| 9 | `boundary_detection` | `qwen-turbo` | 1174.33 | 632 | 82 | 714 |
| 10 | `boundary_detection` | `qwen-turbo` | 1072.68 | 678 | 81 | 759 |
| 11 | `boundary_detection` | `qwen-turbo` | 1109.70 | 725 | 82 | 807 |
| 12 | `boundary_detection` | `qwen-turbo` | 1288.20 | 770 | 115 | 885 |
| 13 | `summary` | `tongyi-xiaomi-analysis-pro` | 712.61 | 204 | 29 | 233 |
| 14 | `role_extraction` | `qwen-turbo` | 5247.57 | 490 | 516 | 1006 |
| 15 | `summary` | `tongyi-xiaomi-analysis-pro` | 746.04 | 81 | 32 | 113 |
| 16 | `summary` | `tongyi-xiaomi-analysis-pro` | 881.03 | 296 | 39 | 335 |
| 17 | `summary` | `tongyi-xiaomi-analysis-pro` | 2723.47 | 331 | 39 | 370 |
| 18 | `summary` | `tongyi-xiaomi-analysis-pro` | 859.82 | 357 | 39 | 396 |
| 19 | `summary` | `tongyi-xiaomi-analysis-pro` | 835.21 | 214 | 37 | 251 |
| 20 | `summary` | `tongyi-xiaomi-analysis-pro` | 886.15 | 203 | 38 | 241 |
| 21 | `summary` | `tongyi-xiaomi-analysis-pro` | 653.23 | 204 | 25 | 229 |
| 22 | `role_extraction` | `qwen-turbo` | 4921.94 | 500 | 502 | 1002 |
| 23 | `summary` | `tongyi-xiaomi-analysis-pro` | 1293.99 | 91 | 67 | 158 |
| 24 | `summary` | `tongyi-xiaomi-analysis-pro` | 890.19 | 208 | 36 | 244 |
| 25 | `role_extraction` | `qwen-turbo` | 5344.07 | 494 | 489 | 983 |
| 26 | `summary` | `tongyi-xiaomi-analysis-pro` | 1030.91 | 85 | 40 | 125 |
| 27 | `summary` | `tongyi-xiaomi-analysis-pro` | 789.48 | 205 | 35 | 240 |
| 28 | `role_extraction` | `qwen-turbo` | 3650.12 | 491 | 343 | 834 |
| 29 | `summary` | `tongyi-xiaomi-analysis-pro` | 872.67 | 82 | 39 | 121 |
| 30 | `summary` | `tongyi-xiaomi-analysis-pro` | 869.19 | 208 | 38 | 246 |
| 31 | `summary` | `tongyi-xiaomi-analysis-pro` | 784.57 | 204 | 34 | 238 |
| 32 | `role_extraction` | `qwen-turbo` | 5052.15 | 494 | 493 | 987 |
| 33 | `summary` | `tongyi-xiaomi-analysis-pro` | 1098.50 | 85 | 57 | 142 |
| 34 | `summary` | `tongyi-xiaomi-analysis-pro` | 712.22 | 205 | 32 | 237 |
| 35 | `role_extraction` | `qwen-turbo` | 4614.61 | 491 | 483 | 974 |
| 36 | `summary` | `tongyi-xiaomi-analysis-pro` | 863.89 | 82 | 41 | 123 |
| 37 | `summary` | `tongyi-xiaomi-analysis-pro` | 765.50 | 205 | 32 | 237 |
| 38 | `role_extraction` | `qwen-turbo` | 4541.41 | 491 | 483 | 974 |
| 39 | `summary` | `tongyi-xiaomi-analysis-pro` | 831.13 | 82 | 39 | 121 |
| 40 | `summary` | `tongyi-xiaomi-analysis-pro` | 832.43 | 382 | 39 | 421 |
| 41 | `summary` | `tongyi-xiaomi-analysis-pro` | 849.40 | 407 | 39 | 446 |
| 42 | `summary` | `tongyi-xiaomi-analysis-pro` | 1057.45 | 433 | 48 | 481 |
| 43 | `summary` | `tongyi-xiaomi-analysis-pro` | 1027.00 | 442 | 48 | 490 |
| 44 | `summary` | `tongyi-xiaomi-analysis-pro` | 1175.73 | 442 | 48 | 490 |
| 45 | `summary` | `tongyi-xiaomi-analysis-pro` | 1041.63 | 442 | 48 | 490 |
| 46 | `summary` | `tongyi-xiaomi-analysis-pro` | 1225.43 | 442 | 48 | 490 |
| 47 | `summary` | `tongyi-xiaomi-analysis-pro` | 1040.75 | 442 | 48 | 490 |
| 48 | `summary` | `tongyi-xiaomi-analysis-pro` | 571.62 | 166 | 20 | 186 |
| 49 | `summary` | `tongyi-xiaomi-analysis-pro` | 547.89 | 166 | 20 | 186 |
| 50 | `summary` | `tongyi-xiaomi-analysis-pro` | 1062.53 | 171 | 49 | 220 |
| 51 | `summary` | `tongyi-xiaomi-analysis-pro` | 564.19 | 206 | 16 | 222 |
| 52 | `summary` | `tongyi-xiaomi-analysis-pro` | 522.11 | 206 | 16 | 222 |
| 53 | `summary` | `tongyi-xiaomi-analysis-pro` | 1019.57 | 238 | 52 | 290 |
| 54 | `summary` | `tongyi-xiaomi-analysis-pro` | 507.59 | 231 | 16 | 247 |
| 55 | `summary` | `tongyi-xiaomi-analysis-pro` | 504.71 | 231 | 16 | 247 |
| 56 | `summary` | `tongyi-xiaomi-analysis-pro` | 1225.95 | 274 | 60 | 334 |
| 57 | `abstraction` | `MiniMax-M2.5` | 22165.52 | 721 | 1275 | 1996 |
| 58 | `summary` | `tongyi-xiaomi-analysis-pro` | 1546.07 | 306 | 82 | 388 |
| 59 | `summary` | `tongyi-xiaomi-analysis-pro` | 4124.91 | 247 | 69 | 316 |
| 60 | `summary` | `tongyi-xiaomi-analysis-pro` | 1438.19 | 235 | 61 | 296 |
| 61 | `summary` | `tongyi-xiaomi-analysis-pro` | 1246.28 | 227 | 61 | 288 |
| 62 | `summary` | `tongyi-xiaomi-analysis-pro` | 1345.51 | 227 | 61 | 288 |
| 63 | `abstraction` | `MiniMax-M2.5` | 17721.28 | 668 | 969 | 1637 |
| 64 | `summary` | `tongyi-xiaomi-analysis-pro` | 1400.56 | 297 | 75 | 372 |
| 65 | `summary` | `tongyi-xiaomi-analysis-pro` | 1206.68 | 241 | 54 | 295 |
| 66 | `summary` | `tongyi-xiaomi-analysis-pro` | 2579.74 | 220 | 37 | 257 |
| 67 | `summary` | `tongyi-xiaomi-analysis-pro` | 1013.27 | 203 | 34 | 237 |
| 68 | `abstraction` | `MiniMax-M2.5` | 22376.36 | 616 | 1089 | 1705 |
| 69 | `summary` | `tongyi-xiaomi-analysis-pro` | 1611.56 | 306 | 90 | 396 |
| 70 | `summary` | `tongyi-xiaomi-analysis-pro` | 1086.80 | 255 | 56 | 311 |
| 71 | `summary` | `tongyi-xiaomi-analysis-pro` | 885.04 | 222 | 41 | 263 |
| 72 | `summary` | `tongyi-xiaomi-analysis-pro` | 704.17 | 207 | 31 | 238 |
| 73 | `abstraction` | `MiniMax-M2.5` | 18082.24 | 566 | 987 | 1553 |
| 74 | `summary` | `tongyi-xiaomi-analysis-pro` | 1150.38 | 250 | 59 | 309 |
| 75 | `summary` | `tongyi-xiaomi-analysis-pro` | 1050.01 | 225 | 52 | 277 |
| 76 | `summary` | `tongyi-xiaomi-analysis-pro` | 873.16 | 218 | 42 | 260 |
| 77 | `summary` | `tongyi-xiaomi-analysis-pro` | 908.77 | 208 | 42 | 250 |
| 78 | `summary` | `tongyi-xiaomi-analysis-pro` | 909.20 | 208 | 42 | 250 |
| 79 | `boundary_detection` | `qwen-turbo` | 904.51 | 566 | 75 | 641 |
| 80 | `abstraction` | `MiniMax-M2.5` | 21153.29 | 860 | 1293 | 2153 |
| 81 | `summary` | `tongyi-xiaomi-analysis-pro` | 1012.70 | 306 | 49 | 355 |
| 82 | `summary` | `tongyi-xiaomi-analysis-pro` | 906.46 | 215 | 42 | 257 |
| 83 | `summary` | `tongyi-xiaomi-analysis-pro` | 759.61 | 208 | 34 | 242 |

**Token 合计（仅统计 API 返回了对应字段的调用）**：prompt Σ=26416 · completion Σ=12742 · total Σ=39158

## 环境 & 复现命令

- **生成时间**：2026-04-18 13:55:51 UTC
- **Python**：`C:\Users\40575\anaconda3\envs\py3125\python.exe`
- **pytest 命令**：`python -m pytest tests/test_wp_live_multi_and_modes.py -v -s`
- **相关环境变量键**（值不落盘）：`REMS_LIVE_ALLOW_FAKE_EMBEDDING`、`REMS_RUN_LIVE_METABOLISM_TEST`、`REMS_WP_REPORT_DIR`

<!-- wp-live-stats: {"report": "wp_live_multi_and_modes.md", "pass": 7, "fail": 0, "na": 0} -->
