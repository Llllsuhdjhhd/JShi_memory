# JShi_memory

JShi_memory 是匠石（JShi）使用的记忆后端项目。Python 包名为 `rems`。当前代码提供记忆写入、回忆、对象肖像查询和 HTTP API；记忆事件、对象关联、检索索引和回忆轨迹由本项目管理。

## 文档入口

- [项目总览](docs/project-overview.md)：运行组成、边界和主要数据流。
- [文档索引](docs/README.md)：当前文档规则、模块地图和历史文档入口。
- [记忆写入](docs/memory.md)：输入模型、代谢封存、事件和存储。
- [回忆检索](docs/recall.md)：过滤、语义/词面检索、近因与长期通道、排序和预算。
- [匠石交互](docs/jshi-integration.md)：MemoryBackendPort、字段映射和 HTTP 接口。
- [开发与验证](docs/development.md)：环境、配置和测试入口。
- [LoCoMo 匠石评测](docs/locomo-evaluation.md)：真实核心、英文记忆、答案条目覆盖及费用控制。
- [草稿箱](docs/drafts/README.md)：模块方案、实施步骤和调试记录的工作区。
- [历史文档](docs/history/README.md)：本次迁移前的文档原件，供追溯，不作为当前行为依据。

## 快速认识

写入入口是 `MemoryBackendPort.ingest_batch`，主要实现位于 `src/rems/pipeline.py`；回忆入口是 `MemoryBackendPort.recall`，检索实现位于 `src/rems/recall/pipeline.py`。FastAPI 路由位于 `src/rems/api/`。

代码和当前契约是行为依据。接口声明、实现、配置和文档之间若有差异，应先记录差异，再以正在运行的实现确认实际行为；不要仅凭历史设计文档推断已上线功能。
