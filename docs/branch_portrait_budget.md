# feat/portrait-recall-budget 分支改动说明（REMS 侧）＋ main 接入指引

> 分支：`Jshi_memory`(REMS 后端) 的 `feat/portrait-recall-budget`。
> 基于上一个提交 `6042110`（召回轨迹 `recall_traces` ＋ 说话人归属 `interlocutor` ＋ 有界软纠偏），**新增两块**：
> ① 人物肖像（per-object 10 级渐进人物摘要 + 记忆疲态）② 回忆预算管理（6 级等级 + 贪心分配）。
> 本文档写给 `main`（匠石 / Jshi 应用）参照，把不涉及的改动接过来。

---

## 一、对外形态（REMS 侧）

### A. 说话人归属 `interlocutor`（此前已提交，main 已接）
- `MemoryExperience.interlocutor`：本段**说话/互动对象** id（区别于 `objects` 泛提及）。
- 已在 `main` 的 `contracts.py` / `memorycontrol/inprocess.py` / `rems3.py` 接好：
  `segment.actor_object_id → MemoryExperience.interlocutor → rems`.
- REMS ingest 会把 `interlocutor` 落到事件 `recall_metadata.interlocutor`，召回 `_object_affinity` 按它软纠偏。

### B. 人物肖像（人物肖像能力）
- **端口**：`REMSPipeline.portrait(subject_id, object_id) -> dict | None`
  （`MemoryBackendPort` 协议已新增 `portrait`）。
- **返回**：`{subject_id, object_id, name, max_level, total_content_len, compression_ratio,
  fatigue, visible_summary, levels:{L1:..}, updated_at}`。
- **数据**：`object_portraits` ＋ `portrait_summaries` 表；模型 `ObjectPortrait` / `PortraitLevelInfo` / `PendingPortraitSummary`。
- **触发**：ingest 时 `PortraitService.note_event` 为**事件涉及的每个对象**各生成一条人物摘要；
  某对象「待并入」摘要累计 ≥ `context_window / 600` 时自动建/更新其肖像。多对象各自独立。
- **等级**：10 级，`L1=max(魔法数100, 总长/2^9)`、逐级×2、超总长即停、最多 10 级；压缩率=总长/最长级。
  **项目统一**：`L1=最简最短`，高级别=更详细更长。
- **两种压缩**：`incremental`（已有最长级+新摘要）/ `full`（所有摘要）；`auto` 时累计总长 ≤ `context_window/6` 用 full，否则 incremental。
- **记忆疲态**（两个值，不共用）：`event_fatigue`（事件，仅整体负担，**与遗忘率协同**：乘进 `effective_factor`）；
  `portrait_fatigue`（肖像，整体+对象合成）；`<1` 时按概率让部分内容概率不可见（期望可见占比≈fatigue，保留偶然性）。
- **配置**：`REMS_PORTRAIT_*`（`magic_num=100`、`max_levels=10`、`growth=2`、`trigger_divisor=600`、
  `compose_mode=auto`、`portrait_fatigue=1.0`、`event_fatigue=1.0`、`enabled`）。

### C. 回忆预算管理
- `RecallBudgetManager`：6 级等级 `L1..L6 = 100/200/400/800/1600/3200`（魔法数 100 起步、×2）。
  **贪心分配**：条目按评分排序，能装最高级装最高级，装不下下降级，再装不下丢弃；实际展示 = `min(档位上限, 内容长度)`。
- **`REMSPipeline.recall()` 返回时已应用预算**（会 `model_copy` 截断 content 到档位、丢弃尾部）；
  预算 = `recall_budget_chars`（0→`physical_redline`），并**把对话人 `object_id` 的人物肖像作为一个高优先条目同场**（`score=100`）。
- **`assemble_recall_block(subject_id, query, *, object_id, level=1, limit=None, anchor_event_ids=(), budget=None)`**
  → `{items:[{event_id, object_id, interlocutor, kind, level, budget, used, content, score}], total_length, budget}`。
- **配置**：`REMS_RECALL_BUDGET_*`（`enabled`、`budget_chars=0`、`level_count=6`、`magic_num=100`、`level_growth=2`、`include_portrait=True`）。

---

## 二、main（Jshi 应用）需要做的改动

### 1. 暴露 `portrait`（如果要给 09/上层用）
`main/src/jshi/memory/port.py` 的 `MemoryBackendPort` 增加：
```python
def portrait(self, subject_id: str, object_id: str) -> dict | None: ...
```
`main/src/jshi/memory/rems3.py` 的 `Rems3MemoryBackend` 增加（同进程 REMSPipeline 已实现 `portrait`）：
```python
def portrait(self, subject_id: str, object_id: str) -> dict | None:
    return self._pipeline.portrait(subject_id, object_id)
```
（`main/src/jshi/memory/backend.py` 的 `InProcessMemoryBackend` 若也走该端口，同样补一个占位/转发。）

### 2. 回忆预算（recall 端已内置，**通常无需改 main**）
- `Rems3MemoryBackend.recall(...)` 调 `pipeline.recall(...)` 即返回**预算受限**的片段（含对话人肖像），直接可用。
- 若想显式拿"块"，可加一道转发方法 `assemble_recall_block(...)` 调 `self._pipeline.assemble_recall_block(...)`。

### 3. 说话人归属 `interlocutor`（**已接好**，确认即可）
- 此三步此前已在 main 加过：`contracts.MemoryExperience.interlocutor`、
  `memorycontrol.inprocess.build_batch` 传 `segment.actor_object_id`、`rems3.experience_payload` 透传 `interlocutor`。
- 若没跟上，请补齐这三处（见下）。

### 4. 配置项（REMS 侧，main 不强制）
- 若要调肖像/预算阈值，在 Jshi 应用启动环境设置 `REMS_PORTRAIT_*` / `REMS_RECALL_BUDGET_*`
  （REMSConfig 读 .env 或环境变量）。

---

## 三、需要你补齐的 main 三处（若此前未加）

`main/src/jshi/memory/contracts.py`（`MemoryExperience`）：
```python
interlocutor: str | None = None   # 说话/互动对象 id；None = 主体自述/系统段
```
`main/src/jshi/memorycontrol/inprocess.py`（`build_batch` 构造 `MemoryExperience`）：
```python
interlocutor=getattr(segment, "actor_object_id", None),
```
`main/src/jshi/memory/rems3.py`（`experience_payload`）：
```python
"interlocutor": getattr(experience, "interlocutor", None),
```

---

## 四、注意

- 这些是 **REMS 后端**改动；`main` 只是消费方，需 `import rems` 的代码工作在该分支（或安装该分支）。
- 本分支改动**未提交**；此前 main 的 3 处 `interlocutor` 接线也**未提交**。
- `PortraitService._obj_name`（肖像名）暂返回 `None`——名字需要从 01 的 `objects` 映射 / 适配器传入（后续）。
- `note_event` 目前按 ingest 同步调用（设计为后台/并行候选）；若需真并行，后续包一层后台 job。
