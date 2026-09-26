# WP-LIVE · 抽象与回忆专项
- **报告文件**：`wp_live_abstraction_and_recall.md`

## 配置快照

```json
{
  "context_window": 8192,
  "chars_per_token": 1.0,
  "len_msg": 124,
  "physical_redline": 1241,
  "safe_watermark": 248,
  "user_mode": "single",
  "core_user_role_id": "ROL-abs-core",
  "active_participants": [],
  "recall_cluster_threshold": 3,
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

## 指标核对表

| 指标 | 状态 | 实测摘要 | 章节 |
| --- | --- | --- | --- |
| §3.2 召回聚集触发 | **PASS** | {"recalled": 4, "produced": true} | §3.2 |
| §1.1.8/§3.1 抽象层级与证据链 | **PASS** | {"abstraction_level": 1, "source_events_count": 5, "cluster_size": 5} | §3.2 |
| §3.3 防幻觉锚定开关 | **PASS** | {"on_has_anchor": true, "off_no_anchor": true} | §3.3 |
| §4.3 墓碑回忆排除 | **PASS** | {"recall_ids": ["EVT-0019da0e4f5e9-fecaa725", "EVT-0019da0e48fdc-2f1e35c0", "EVT-0019da0e443a9-4615ec5e", "EVT-0019da0e44360-7998cb0f", "EVT-0019da0e4a93a-6c10b98a", "EVT-0019da0e4a8be-9952ee22", "EVT | §4.3 |
| §4.4 回忆块 ≤ physical_redline | **PASS** | 116 | §4.4 |
| §4.4 ultra fallback 触发 | **FAIL** | {"levels": ["L2", "L2", "L2", "L2"], "has_ultra": false} | §4.4 |

## §3.2 召回聚集触发

### 白皮书原文

> 近邻基本事件数达到阈值时应归纳抽象事件。
### 检查方法

- 手造同主题事件 + `check_and_abstract`。
### 现场数据 · §3.2 召回聚集触发 **`PASS`**

- **期望**：recalled ≥ 3 ⇒ produced=True；否则 produced=False
- **实测**：`{"recalled": 4, "produced": true}`
### 抽象事件摘录

```json
{
  "event_id": "EVT-0019da0e48fdc-2f1e35c0",
  "abstraction_level": 1,
  "source_events": [
    "EVT-0019da0e4423a-a4d270bd",
    "EVT-0019da0e443a9-4615ec5e",
    "EVT-0019da0e44360-7998cb0f",
    "EVT-0019da0e442af-4880e12c",
    "EVT-0019da0e44385-e232ec32"
  ],
  "insight": "项目进度会议遵循「目标确认→风险预判→资源协调→问题复盘→计划对齐」的递进逻辑。该模式体现了项目管理中的PDCA循环意识：先明确方向（里程碑），再识别风险（风险清单），接着解决人的问题（接口人协调），然后从历史问题中学习（延期复盘），最后落地执行细节（测试计划对齐）。记录编号的顺序性揭示了成熟的项目管理流程具备可复制的结构化框架。"
}
```
### 现场数据 · §1.1.8/§3.1 抽象层级与证据链 **`PASS`**

- **期望**：abstraction_level == 1, source_events ⊇ cluster - self
- **实测**：`{"abstraction_level": 1, "source_events_count": 5, "cluster_size": 5}`


```
{
  "event_id": "EVT-0019da0e48fdc-2f1e35c0",
  "missing_sources": []
}
```

## §3.3 防幻觉锚定开关

### 白皮书原文

> 锚定概率强制为 1 时用户提示应出现 L1 锚定标签；为 0 时不出现。
### 检查方法

- 两次 `check_and_abstract`，比较 abstraction user 消息。
### 现场数据 · §3.3 防幻觉锚定开关 **`PASS`**

- **期望**：on 时出现 L1（锚定事实层），off 时不出现
- **实测**：`{"on_has_anchor": true, "off_no_anchor": true}`
### 锚定 ON 消息头

```json
"## 子事件摘要集合（共 5 个事件）\n### 事件 EVT-0019da0e4423a-a4d270bd (创建于 2026-04-18T22:00:06.970980) [L1（锚定事实层）]\n张三在会议室讨论项目进度：梳理风险清单（记录-0）。\n\n### 事件 EVT-0019da0e443a9-4615ec5e (创建于 2026-04-18T22:00:07.337845) [L1（锚定事实层）]\n张三在会议室讨论项目进度：对齐测试计划（记录-4）。\n\n### 事件 EVT-0019da0e44360-7998cb0f (创建于 2026-04-18T22:00:07.264118) [L1（锚定事实层）]\n张三在会议室讨论项目进度：协调接口人（记录-2）。\n\n### 事件 EVT-0019da0e442af-4880e12c (创建于 2026-04-18T22:00:07.087359) [L1（锚定事实层）]\n张三在会议室讨论项目进度：确认里程碑日期（记录-1）。\n\n### 事件 EVT-0019da0e44385-e232ec32 (创建于 2026-04-18T22:00:07.301270) [L1（锚定事实层）]\n张三在会议室讨论项目进度：回顾上周延期原因（记录-3）。\n\n返回 JSON：\n```json\n{\n  \"content_raw\": \"合成事实描述\",\n  \"insight\": \"规律与见解\",\n  \"decoration\": \"主观装饰（可选）\",\n  \"roles\": [\n    {\n      \"role_id\": \"...\",\n      \"importance\": \"S|A|B|C|D\",\n      \"l3_decision\": \"典型决策风格描述\",\n      \"emotion_trend\": {\n        \"vedana\": {},\n        \"klesha\": {}\n      }\n    }\n  ]\n}\n```"
```
### 锚定 OFF 消息头

```json
"## 子事件摘要集合（共 7 个事件）\n### 事件 EVT-0019da0e4a8be-9952ee22 (创建于 2026-04-18T22:00:33.214856) [L2]\n张三在会议室讨论项目进度：B-补充预算评估（记录-10）。\n\n### 事件 EVT-0019da0e4a93a-6c10b98a (创建于 2026-04-18T22:00:33.339100) [L2]\n张三在会议室讨论项目进度：B-记录决策待办（记录-13）。\n\n### 事件 EVT-0019da0e443a9-4615ec5e (创建于 2026-04-18T22:00:07.337845) [L2]\n张三在会议室讨论项目进度：对齐测试计划（记录-4）。\n\n### 事件 EVT-0019da0e4a913-60bb2027 (创建于 2026-04-18T22:00:33.299996) [L2]\n张三在会议室讨论项目进度：B-确认外包范围（记录-12）。\n\n### 事件 EVT-0019da0e4a8e7-ed1848cf (创建于 2026-04-18T22:00:33.255023) [L2]\n张三在会议室讨论项目进度：B-讨论资源瓶颈（记录-11）。\n\n### 事件 EVT-0019da0e44360-7998cb0f (创建于 2026-04-18T22:00:07.264118) [L2]\n张三在会议室讨论项目进度：协调接口人（记录-2）。\n\n### 事件 EVT-0019da0e442af-4880e12c (创建于 2026-04-18T22:00:07.087359) [L2]\n张三在会议室讨论项目进度：确认里程碑日期（记录-1）。\n\n返回 JSON：\n```json\n{\n  \"content_raw\": \"合成事实描述\",\n  \"insight\": \"规律与见解\",\n  \"decoration\": \"主观装饰（可选）\",\n  \"roles\": [\n    {\n      \"role_id\": \"...\",\n      \"importance\": \"S|A|B|C|D\",\n      \"l3_decision\": \"典型决策风格描述\",\n      \"emotion_trend\": {\n        \"vedana\": {},\n        \"klesha\": {}\n      }\n    }\n  ]\n}\n```"
```

## §4.3 墓碑排除

### 白皮书原文

> 墓碑事件不得出现在回忆检索结果中。
### 检查方法

- `tombstone` 后 `build_recall_block`。
### 现场数据 · §4.3 墓碑回忆排除 **`PASS`**

- **期望**：回忆块不含 EVT-0019da0e518ec-5e9d46a3
- **实测**：`{"recall_ids": ["EVT-0019da0e4f5e9-fecaa725", "EVT-0019da0e48fdc-2f1e35c0", "EVT-0019da0e443a9-4615ec5e", "EVT-0019da0e44360-7998cb0f", "EVT-0019da0e4a93a-6c10b98a", "EVT-0019da0e4a8be-9952ee22", "EVT`
**墓碑前 ID 预览**

- `EVT-0019da0e4f5e9-fecaa725`
- `EVT-0019da0e48fdc-2f1e35c0`
- `EVT-0019da0e443a9-4615ec5e`
- `EVT-0019da0e44360-7998cb0f`
- `EVT-0019da0e4a93a-6c10b98a`
- `EVT-0019da0e4a8be-9952ee22`
- `EVT-0019da0e4a913-60bb2027`
- `EVT-0019da0e44385-e232ec32`

**墓碑后 ID 预览**

- `EVT-0019da0e4f5e9-fecaa725`
- `EVT-0019da0e48fdc-2f1e35c0`
- `EVT-0019da0e443a9-4615ec5e`
- `EVT-0019da0e44360-7998cb0f`
- `EVT-0019da0e4a93a-6c10b98a`
- `EVT-0019da0e4a8be-9952ee22`
- `EVT-0019da0e4a913-60bb2027`
- `EVT-0019da0e44385-e232ec32`

## §4.4 ultra 与物理红线

### 白皮书原文

> 回忆块总长不得超过 `physical_redline`，必要时 ultra 极简降级。
### 检查方法

- 窄上下文 + 超长 L1 触发 `_ultra_concise_fallback`。
### 现场数据 · §4.4 回忆块 ≤ physical_redline **`PASS`**

- **期望**：≤ 193 字
- **实测**：`116`


```
{
  "items": 4,
  "levels": [
    "L2",
    "L2",
    "L2",
    "L2"
  ]
}
```
### 现场数据 · §4.4 ultra fallback 触发 **`FAIL`**

- **期望**：items 中至少一条 summary_level == 'ultra'
- **实测**：`{"levels": ["L2", "L2", "L2", "L2"], "has_ultra": false}`

## §4.4 混合打分分量

### 白皮书原文

> 混合分由余弦、时间衰减、角色 boost、AE、activation_energy 组成。
### 检查方法

- `install_recall_scoring_tracer` 记录分项贡献。
```text
`EVT-0019da0e4a93a-6c10b98a` cosine:0.304 | time:0.150 | role:0.040 | ae:0.000 | act:0.000 → total=0.494
`EVT-0019da0e4a8be-9952ee22` cosine:0.290 | time:0.150 | role:0.040 | ae:0.000 | act:0.000 → total=0.480
`EVT-0019da0e4a913-60bb2027` cosine:0.290 | time:0.150 | role:0.040 | ae:0.000 | act:0.000 → total=0.480
`EVT-0019da0e4a8e7-ed1848cf` cosine:0.284 | time:0.150 | role:0.040 | ae:0.000 | act:0.000 → total=0.474
```

## 运行摘要

- 抽象事件生成：**EVT-0019da0e48fdc-2f1e35c0**
- 墓碑 event_id：`EVT-0019da0e518ec-5e9d46a3`

## LLM 调用记录

数据来自 **`LLMProvider.invocation_history()`**（每次 ``complete`` 一行）。

**延迟**为客户端测量的往返时间（毫秒）；**token** 取自 API `usage`，网关未返回时显示为「—」。

| # | task_type | model | 延迟(ms) | prompt | completion | total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `abstraction` | `MiniMax-M2.5` | 19447.12 | 552 | 693 | 1245 |
| 2 | `summary` | `tongyi-xiaomi-analysis-pro` | 1416.83 | 284 | 58 | 342 |
| 3 | `summary` | `tongyi-xiaomi-analysis-pro` | 1242.90 | 224 | 45 | 269 |
| 4 | `summary` | `tongyi-xiaomi-analysis-pro` | 1228.04 | 211 | 43 | 254 |
| 5 | `summary` | `tongyi-xiaomi-analysis-pro` | 1247.56 | 209 | 37 | 246 |
| 6 | `summary` | `tongyi-xiaomi-analysis-pro` | 1141.88 | 203 | 37 | 240 |
| 7 | `abstraction` | `MiniMax-M2.5` | 19575.22 | 649 | 800 | 1449 |
| 8 | `summary` | `tongyi-xiaomi-analysis-pro` | 1993.49 | 295 | 86 | 381 |
| 9 | `summary` | `tongyi-xiaomi-analysis-pro` | 1844.42 | 252 | 85 | 337 |
| 10 | `summary` | `tongyi-xiaomi-analysis-pro` | 3923.81 | 251 | 55 | 306 |
| 11 | `summary` | `tongyi-xiaomi-analysis-pro` | 1100.15 | 221 | 32 | 253 |

**Token 合计（仅统计 API 返回了对应字段的调用）**：prompt Σ=3351 · completion Σ=1971 · total Σ=5322

## 环境 & 复现命令

- **生成时间**：2026-04-18 14:00:06 UTC
- **Python**：`C:\Users\40575\anaconda3\envs\py3125\python.exe`
- **pytest 命令**：`python -m pytest tests/test_wp_live_abstraction_and_recall.py -v -s`
- **相关环境变量键**（值不落盘）：`REMS_LIVE_ALLOW_FAKE_EMBEDDING`、`REMS_RUN_LIVE_METABOLISM_TEST`、`REMS_WP_REPORT_DIR`

<!-- wp-live-stats: {"report": "wp_live_abstraction_and_recall.md", "pass": 5, "fail": 1, "na": 0} -->
