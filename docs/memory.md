# 记忆写入与存储

本文按当前 `src/rems/` 实现描述写入行为。

## 输入模型

`MemoryExperience` 表示一段经历，主要字段如下：

| 字段 | 当前用途 |
|---|---|
| `subject_id` | 匠石实例作用域；同一批经历必须属于该主体 |
| `text` | 经历原文，写入层不主动改写；内部代谢可以分析和封装事件 |
| `objects` | 对象称呼到对象 ID 的映射，可包含未在原文中出现或未说话的对象 |
| `interlocutor` | 该历史经历段的对话/互动对象 ID，不表示当前回忆查询的对话人 |
| `source_ids` | 事实、活动或认知等来源 ID |
| `occurred_at` | 经历发生时间；为空时事件时间回退到摄入时间 |
| `segment_id` | 匠石经历段标识，用于写入回执关联 |
| `origin` | `external`、`internal`、`dream` 等来源标签 |

`MemoryBatch` 按顺序携带多段经历。空文本会记入 `errors` 并跳过；当前批次中剩余有效经历会以空行拼接为一次代谢输入。合并对象和来源集合的同时，`SubSegment` 会保留每段原文、对象映射、段 ID 和历史对话对象，以免只用整批对象并集而造成事件归属串扰。

## 写入主流程

1. `REMSPipeline.ingest_batch` 校验并合并有效经历。
2. `MetabolismService` 处理输入边界和未封存内容。
3. `EventService` 为封存事件生成/保存原文、摘要、角色与情绪等事件信息。
4. 已封存事件写入事件仓储，并由 `RecallPipeline.index_event` 为事件 L1 和对象事实建立检索索引。
5. `stored_marks` 将匠石 `segment_id` 对应到本轮封存的事件 ID；尚未封存的内容通过 `unclosed_count` 反映。

一次输入不保证立刻形成一条封存事件：边界检测和代谢会决定如何封存，未完成内容会保留在未封存存储中。

## 事件与对象归属

事件由 `Event` 模型承载，包含 `content_raw`、多级 `summaries`、主体 ID、来源 ID、发生/创建时间、对象角色快照和 `interlocutor_attributions` 等信息。历史对话人归属按来源经历段记录；批次级单一 `interlocutor` 不能替代段级归属。

记忆模型还包含基本/抽象事件标记、事件状态和墓碑字段。`tombstone` 是逻辑标记路径；它用于让事件在审计数据中保留并从常规回忆中排除。不要把墓碑化与物理删除混为一谈。

## 当前存储

表定义见 `src/rems/storage/database.py`。常见表包括：

- `events`：事件主体、原文、摘要、时间、状态及生命周期字段。
- `roles`、`white_painting_entries`、`object_memory_entries`：对象/角色信息及其与记忆材料的关联。
- `event_tier1`：事件一级摘要等检索文本。
- `unclosed_events`、`shadow`：未封存事件和代谢上下文数据。
- `stored_marks`：输入段到封存事件的写入回执关系。
- `recall_traces`：启用时记录的查询和命中观测信息；不作为事件生成输入。
- `object_portraits` 及相关对象画像表：启用画像服务时保存的对象画像数据。

向量集合名由基础集合名与 `recall_embedding_version` 组合。更换嵌入模型/维度时，应评估版本号并重建索引，避免不同向量空间的数据混用。

## 遗忘相关的写入字段

事件保存 `activation_energy` 和 `forgetting_factor`。当前事件服务会从情绪唤醒度得到激活能量，并以配置 `recall_activation_gain` 计算初始遗忘因子；回忆阶段的时间衰减和命中强化见[回忆机制](recall.md#可及性近期通道与长期通道)。当前这套机制影响可检索概率和排序，不意味着系统按该因子定期物理删除事件。
