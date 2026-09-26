# 810 存储与 Schema

> 范围：810–1009（子文档可用 820–1000）｜状态：已确认（定稿）

## 1. 改动内容

### 1.1 现状（REMS3 表）

- `events`：含抽象字段（is_abstract / abstraction_level / source_events / abstract_coverage 等）、role_list、无 subject_id、origin=normal / dream；
- `roles` + `white_painting_entries`：角色 + 白描（含情感与遗忘因子）；
- `shadow` / `unclosed_events`：残影 / 未完成（内部缓冲）；
- `recall_log` / `abstracted_subsets`：抽象触发；
- `event_tier1`（A-Res 活跃池）与 Qdrant 向量索引：与回忆相关；
- 无 `stored_marks`、无对象时间线、无事件级主体情感字段。

### 1.2 目标 Schema

**events（记忆单元）**

| 字段 | 说明 |
|------|------|
| `memory_event_id` | 主键（现 event_id） |
| `subject_id` | 归属匠石，恒在；索引 |
| `content_raw` | 原文，永不改写 |
| `object_ids` | JSON list（01 object_id；空 = 主体记忆） |
| `summaries` | JSON（L1–L10 多级） |
| `source_ids` | JSON list（来源链） |
| `occurred_at` | 经历时间（`create_time` 保留为摄入时间） |
| `origin` | external / internal / dream；**字符串、保留扩展**（不硬校验三值） |
| `location` | 地点（事件要素：时间 / 人物 / 地点） |
| `emotion` | 事件级主体情感 JSON（8 维 + arousal / valence / energy） |
| `memory_status` | sealed / suppressed（对外；内部映射 Event.status） |
| `input_id` | 30 侧经历段 id；索引 |
| 保留 | 80/20 分裂链字段（内部切分不变） |
| 删除 | 抽象字段、role_list 中的对象情感 |

**object_memory_entries（对象时间线，替代 roles + white_painting_entries）**

- `subject_id`、`object_id`、`name`（呈现用）、`event_id`、`summary`（一句摘要）、`create_time`；
- **无情感、无遗忘因子**（遗忘在记忆单元层）；
- 索引：subject_id + object_id + create_time。

**stored_marks（台账）**

- `subject_id`、`input_id`（主键）、`event_ids`（JSON list，封存几个填几个）、`created_at`；
- 纯台账，不含游标状态（30 侧凭 stored_marks 自维护 memory_start）。

**shadow / unclosed_events（内部缓冲）**

- 加 `subject_id` 作用域；不跨端口暴露。

**删除**

- `recall_log`、`abstracted_subsets`（无抽象）；
- `event_tier1`、Qdrant 向量索引：**占位**——本轮不写，回忆启用时再论（届时需重建索引）。

**向量库（Qdrant）占位**

- 配置 / 客户端保留，本轮不建集合、不写向量；与回忆相关，回忆启用时再论。

### 1.3 字段映射

- `origin`：新数据写 `external`；legacy `normal` 读取时归一为 `external`（别名兼容）；
- `memory_status`（**已定**）：`sealed` = 内部 `active`（封存）；`suppressed` = 内部 `silent`（遗忘静默，只降可见性、不删行）；`unclosed` 仅缓冲；
- `emotion`（**已定**）：事件顶层字段（主体唯一；EMA 滚动可由事件历史推导）。

### 1.4 涉及代码

- `storage/database.py`：表定义与迁移（删抽象表、加 subject_id / stored_marks / object_memory_entries / emotion / location）；
- `storage/repository.py`：EventRepository 按 subject_id；对象时间线仓储；stored_marks 仓储；删除 recall_log / abstracted 仓储；
- `storage/vector_store.py` / `qdrant_store.py`：占位（不调用）；
- white_painting_entries：恢复写入——匠石对对象的分级白描（l1/l2/l3，事件关联，无情感、无等级）。

## 2. 原因

- 契约记忆单元字段一一落库；对象来自 01（时间线按 object_id）；原文保全；无抽象；stored_marks 台账；事件要素（时间 / 人物 / 地点）；
- **向量与回忆相关，暂缓**（常理：不为无消费方的索引付成本，回忆启用时再建）；
- 常理原则：suppressed 只降可见性、不删原文；失败不污染历史。

## 3. 目标

- schema 与契约记忆单元一致；subject_id 作用域（恒单主体）；对象时间线无情感；stored_marks 列表台账；无抽象表；
- 向量 / tier1 占位，不写；
- legacy 兼容（normal → external）；
- `tests/unit` 基线不回归。

## 4. 与其他模块的关系

| 模块 | 关系 |
|------|------|
| 210 输入协议 | 输入字段落库（objects / source_ids / occurred_at / origin / input_id） |
| 410 对象时间线 | object_memory_entries（object_id + name + 摘要 + 时间） |
| 610 事件封装 | 封存写库（事件要素、emotion、memory_status、stored_marks） |
| 1010 回忆（预留） | 数据兼容；向量 / 活跃池回忆时再论 |
| 1210 / 1410 | 删表清单（recall_log、abstracted_subsets、roles 改造） |

## 5. 预留的功能

- `origin` 扩展：字符串字段，新值注册（与 30 侧同步）；
- `activity_id`：本轮不落库，schema 预留；
- 向量索引 / `event_tier1`：回忆启用时重建；
- 多主体：`subject_id` 参数化（当前恒单主体）；
- `importance`：对象时间线可扩展（回忆启用时评估）。

## 6. 定稿决策（2026-08-23 确认）

1. **memory_status 映射**：sealed = 内部 active（封存）；suppressed = 内部 silent（遗忘静默，只降可见性、不删行）；unclosed 仅缓冲；
2. **事件级主体情感**：事件顶层 emotion 字段；EMA 滚动由事件历史推导；
3. **stored_marks 纯台账**：不含游标状态，30 侧凭它自维护 memory_start；
4. **location 提取**：由 enrichment 提取落库（沿用 REMS3 keywords / location 机制）；
5. **索引设计**：subject_id + create_time、subject_id + object_id 组合索引；
6. **legacy 处理**：本 fork 无生产数据，抽象字段直接删；normal 读取时归一为 external（不迁移）。

## 7. 验收标准

### 文档阶段

- 已通过评审（2026-08-23），定稿。

### 代码阶段

- schema 与契约记忆单元字段一致；对象时间线无情感；stored_marks 列表台账；origin 三值 + 扩展；
- 向量 / tier1 不写入；删表后无抽象代码 / 表残留；
- `tests/unit` 基线不回归。
