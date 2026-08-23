# Jshi_memory — REMS3 fork 说明（参考文档）

## 一、缘由

匠石（JShi）项目把记忆系统（模块 09）定位为可替换后端：主流程只依赖 MemoryPort 的形状（存储、回忆、整理、做梦），不绑定任何具体引擎。经通读 REMS3 后决定 fork 它作为记忆后端，但必须按匠石的三条硬约束改造：不做抽象事件；不建独立角色系统，对象一律用 01 的 object_id；记忆只做后端，不决定主体面。

REMS3 原目录位于项目\memory\rems_3，当前在 test/hongloumeng 分支，工作树里有一大批尚未提交的核心代码（retrieval、ingest、act_text、query_act_processor、ingest_session 等），同时按约定应保持只读。直接在原目录改动会污染参考源，也有丢失未提交代码的风险。因此 2026-08-19 新建本目录作为独立 fork 工作区。

## 二、目录与仓库状态

- 路径：C:\Users\40575\Desktop\prog\Jshi_memory，独立 git 仓库，分支 main，基线提交 5699c36，180 个文件，已跟踪约 1 MB。
- 复制时排除：.env（密钥）、.git、各类缓存、.vscode/.cursor、logs、scratch、test_chroma、data。红楼梦数据集未带，场景测试当前会失败。
- 原 rems_3 保持只读参考，不再修改；本目录是后续唯一工作区。
- tests 里约 63 MB 的场景输出已被 .gitignore 排除。

## 三、当前构成与已知债务

src/rems 保留全部模块：pipeline 主干、models（Event、Shadow、Unclosed、Recall）、services（代谢、封存、回忆、角色、抽象、信念修订）、skills（边界、摘要、充实、角色抽取、召回意图、做梦）、retrieval（三频段加 BM25 加 RRF）、storage（SQLAlchemy 加 Qdrant）、llm（provider、prompts、session）、strategies、observability、jobs、api。

已知债务：pyproject 引用的 readme.md 已删除；配置默认绑 deepseek 与 Qdrant；整个 schema 没有 subject_id，shadow、unclosed、events、recall_log、tier1 全是全局单主体；抽象链与角色系统与匠石约束冲突。

## 四、改造顺序（按主题提交）

1. schema 与仓储加 subject_id 作用域——涉及全部表与仓储，是最容易被低估的一步。
2. 删抽象链：AbstractionService、InductiveEvolutionSkill、mining、ASF、abstracted_subsets 及 Event 的抽象字段；recall_log 去留单独决定。
3. 角色换 01 对象：召回焦点、role_list、白描时间线全部改用 object_id；角色抽取只返回 unresolved_object_refs 候选，建档与确认归 01。
4. 批次适配：接收 MemoryBatch（new_input、new_response、in_context_recalls）；残影与未完成事件留在后端内部，不跨端口暴露。
5. 配置可注入：LLM 端点与向量后端可替换；删除 DIALOGUE、NPC 输出模式，只留后端端口。

## 五、接回匠石的契约与操作提示

最终以 JShi 09 的 MemoryBackendPort 为准：ingest_batch 返回 BackendIngestResult；recall 返回 BackendRecallResult；对象必须来自 01；禁止抽象事件。集成做薄适配器，JShi 保留 InProcess 后端兜底。

操作提示：conda 用 rems 或 py3125；先跑 tests/unit 建立重构前基线；每完成一步跑对应测试并单独提交；WORK_GUIDE.md 是原项目工作指南，算法与测试约定照它执行。