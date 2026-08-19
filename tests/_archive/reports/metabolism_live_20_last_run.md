# REMS 代谢实况：前 2 次输入

- 生成时间: 2026-04-16 14:05:25 UTC
- 配置摘要: `context_window=32768`, `len_msg=744`, `physical_redline=7447`, `safe_watermark=1489`
- LLM: `base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`（密钥未写入本文档）
- **Chroma 嵌入**: 真实本地模型 `BAAI/bge-small-zh-v1.5`（`torch` 2.4.1 · 解释器: `C:\Users\40575\anaconda3\envs\py3125\python.exe`）

## 测试范围说明

本轮仅调用 **`MetabolismService.process_input`**（不经由 `REMSPipeline.ingest`），因此不会触发：回忆块组装、基于回忆的再巩固抽象、以及封存后的 `check_and_abstract` 向量聚类；也不会执行 **`RoleService.update_from_event`**（白描时间线 / 语义卡片 LLM 刷新）。

仍会完整执行：**边界检测 → 对已闭环片段 `seal_event`（递归摘要、角色抽取、decoration）→ 更新残影与未完成库**。

---

## 第 1 轮输入

### 本轮用户原文

```text
早上八点我出门，天气有点阴。
```

### 本轮新封存的基本事件

_本轮边界模型未产出已闭环片段，故无新 `Event` 入库。_

### 本轮结束后的残影（Shadow）

- **字符数**: 14
- **updated_at**: 2026-04-16 22:05:26.344509

```text
早上八点我出门，天气有点阴。
```

### 本轮结束后的未完成事件库（Unclosed）

- **UC-caf38a4e** · 片段数 1 · 总长 14
  - 合并预览: 早上八点我出门，天气有点阴。

---

## 第 2 轮输入

### 本轮用户原文

```text
在小区门口买了一杯美式咖啡。
```

### 本轮新封存的基本事件

- **EVT-0019d969c8618-c9980a6c** · 状态 `active` · 角色: null, null
  - **L0**（前 200 字）: 早上八点我出门，天气有点阴。 在小区门口买了一杯美式咖啡。
  - **L1**（前 200 字）: 八点出门，天阴；小区门口买美式咖啡。

```json
{
  "event_id": "EVT-0019d969c8618-c9980a6c",
  "create_time": "2026-04-16T22:05:33.592183",
  "content_raw": "早上八点我出门，天气有点阴。\n在小区门口买了一杯美式咖啡。",
  "summaries": {
    "L1": "八点出门，天阴；小区门口买美式咖啡。"
  },
  "summary_lengths": {
    "L1": 16
  },
  "actual_max_level": 1,
  "role_list": [
    {
      "role_id": "null",
      "importance": "S",
      "role_snapshot": {
        "l1_mention": "出现",
        "l2_interaction": "早上八点出门，购买了一杯美式咖啡",
        "l3_decision": "日常行为，无特殊决策"
      },
      "emotional_model": {
        "vedana": {
          "joy": 0.0,
          "suffering": 0.0,
          "happiness": 0.0,
          "worry": 0.0,
          "equanimity": 0.0
        },
        "klesha": {
          "greed": 0.0,
          "anger": 0.0,
          "ignorance": 0.0,
          "pride": 0.0,
          "doubt": 0.0,
          "wrong_view": 0.0
        }
      }
    },
    {
      "role_id": "null",
      "importance": "B",
      "role_snapshot": {
        "l1_mention": "出现",
        "l2_interaction": "被购买",
        "l3_decision": "作为日常饮品被选择"
      },
      "emotional_model": {
        "vedana": {
          "joy": 0.0,
          "suffering": 0.0,
          "happiness": 0.0,
          "worry": 0.0,
          "equanimity": 0.0
        },
        "klesha": {
          "greed": 0.0,
          "anger": 0.0,
          "ignorance": 0.0,
          "pride": 0.0,
          "doubt": 0.0,
          "wrong_view": 0.0
        }
      }
    }
  ],
  "is_abstract": false,
  "is_abstracted": false,
  "status": "active",
  "decoration": "晨光在阴云下晕开，灰白的天幕低垂，仿佛城市的呼吸都慢了下来。我踏出家门，拐角的咖啡香在冷空气中凝成一缕暖意，黑色液体里藏着清晨的沉默与期待。",
  "insight": null,
  "event_length": 29,
  "abstraction_level": null,
  "source_events": null,
  "is_tombstoned": false
}
```

### 本轮结束后的残影（Shadow）

- **字符数**: 14
- **updated_at**: 2026-04-16 22:05:33.717433

```text
早上八点我出门，天气有点阴。
```

### 本轮结束后的未完成事件库（Unclosed）

_（当前无未完成事件）_

---

## 2 轮结束后的全局快照

- **库中基本事件总数**（非抽象、非墓碑）: **1**

### 按 `create_time` 排序的基本事件一览

- **EVT-0019d969c8618-c9980a6c** · 状态 `active` · 角色: null, null
  - **L0**（前 200 字）: 早上八点我出门，天气有点阴。 在小区门口买了一杯美式咖啡。
  - **L1**（前 200 字）: 八点出门，天阴；小区门口买美式咖啡。

### 最终残影

- 字符数: **14**

```text
早上八点我出门，天气有点阴。
```

### 最终未完成事件

_（当前无未完成事件）_

## 轮次摘要表

| 轮次 | 输入字数 | 新封存事件数 | 新事件 ID | 残影字数 | 未完成条数 |
|-----:|---------:|-------------:|-----------|---------:|-----------:|
| 1 | 14 | 0 | — | 14 | 1 |
| 2 | 14 | 1 | EVT-0019d969c8618-c9980a6c | 14 | 0 |
