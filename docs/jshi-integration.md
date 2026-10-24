# 与匠石交互

本仓库通过 Python 端口和可选 HTTP API 对外提供记忆能力。与 JShi 直接协作时，应以两侧端口代码和跨仓契约测试为准。

## Python 端口

`src/rems/port.py` 定义：

- `MemoryExperience` / `MemoryBatch`：输入经历和批次。
- `BackendIngestResult`：写入错误、封存事件 ID、对象 ID、未封存数量和段级 `stored_marks`。
- `RecalledFragment`：返回的回忆材料及对象、来源、分数和历史对话人归属。
- `MemoryBackendPort`：写入、回忆、肖像、巩固和做梦方法的协议形状。

主流程当前有写入、回忆和肖像实现；`consolidate`、`dream` 仍是协议声明，调用前需要确认具体后端是否实现，不能只因协议有方法就视为可用。

## JShi 回忆字段

REMS 的 `recall` 接口支持：

| 参数 | 说明 |
|---|---|
| `subject_id`, `query` | 主体作用域和查询文本 |
| `object_id`, `object_ids` | 单个或多个查询对象 |
| `interlocutor_object_id` | 本轮查询对话人对象 ID；会用于区分历史对话对象和其他涉及对象 |
| `time_range` | 起止时间元组；与 `/api/recall` 同名的文本字段映射略有不同 |
| `budget_chars`, `expand_raw`, `limit` | 字符预算、是否展开原文和片段数量 |
| `level`, `anchor_event_ids` | 为兼容端口而接收。细节条数不随 `level` 变化；原文附在同一条 `content` 里。`detail_event_ids` 随结果返回；预算放不下时整段去掉，不从原文中间截断 |

对象角色权重当前是：历史事件的对话对象 0.7，其他涉及对象 0.3。历史对话对象归属从事件来源子段写入 `interlocutor_attributions`，返回标签表示事件的历史互动关系，不表示该对象一定在回忆材料中亲自说过某句文本。

如果调用方不显式传 `interlocutor_object_id`，`REMSPipeline.recall` 会把 `object_ids` 第一项当作本轮对话对象。这个兼容约定依赖输入顺序；新调用优先显式传对话对象。

`tests/integration/test_jshi_recall_contract.py` 在相邻 `JShi/src` 可用时离线检查 JShi 参数形状、协调器的对象/对话人传递及回忆结果映射。它不请求真实 LLM。

## HTTP API

FastAPI 应用由 `src/rems/api/app.py` 创建，全部路由带 `/api` 前缀：

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/api/ingest` | 批量写入经历 |
| `POST` | `/api/recall` | 查询回忆片段 |
| `GET` | `/api/events` | 按抽象/墓碑条件列出事件 |
| `GET` | `/api/events/{event_id}` | 获取单个事件 |
| `POST` | `/api/events/tombstone` | 对事件设置墓碑 |
| `POST` | `/api/roles/query` | 按名称或 ID 查询对象/角色 |
| `GET` | `/api/roles` | 列出角色 |
| `GET` | `/api/roles/{role_id}/conflicts` | 检查对象信念冲突 |
| `GET` | `/api/health` | 健康检查 |

`POST /api/recall` 可传 `object_ids`、`interlocutor_object_id`、`time_start`、`time_end` 和字符预算。当前只有两个时间端点都存在时才构造时间过滤。时间字符串使用 Python `datetime.fromisoformat` 可解析的格式。

## 变更契约时的核对

修改任一侧回忆字段时，同步核对 `src/rems/port.py`、`src/rems/pipeline.py`、`src/rems/api/routes.py`、JShi 适配代码和 `tests/integration/test_jshi_recall_contract.py`。分清历史对话人字段与本轮查询对话人字段，并验证多对象顺序、时间过滤及结果转换。
