# JShi_memory 测试说明

本仓库为匠石提供可替换的记忆后端。回忆行为以 `design/1010-回忆设计.md` 为准，匠石调用边界以 `src/rems/port.py` 和邻接 JShi 仓库的 `src/jshi/memory/port.py` 为准。

## 测试分层

| 层 | 路径 | 用途 | 是否需要外部服务 |
|---|---|---|---|
| 单元 | `tests/unit/` | 模型、仓储、写入封存、回忆检索和预算 | 否；使用 SQLite 临时库、内存 Qdrant、HashEmbedding 和 FakeLLM |
| 双仓契约 | `tests/integration/test_jshi_recall_contract.py` | 核对 JShi `MemoryPort` 的回忆参数和结果字段转换 | 否；需要 `JShi` 与 `JShi_memory` 两仓并列 |
| 集成 / 场景 | `tests/integration/`、`tests/scenarios/` | 验证需真实模型或较长文本的端到端行为 | 视具体用例而定；不作为离线单元测试前置条件 |
| 归档 | `tests/_archive/` | 留存旧实验和诊断材料 | 不纳入默认 pytest 收集 |

## 建立 Miniconda 测试环境

仓库提供 `environment-test.yml`。在项目根目录运行：

```powershell
conda env create -f environment-test.yml
conda activate jshi-memory-test
```

该环境使用 Python 3.12。单元测试走 `HashEmbedding`，不需要下载 sentence-transformers 模型或连接 LLM。

## 常用命令

```powershell
# 回忆与写入的离线单元测试
pytest tests/unit -q

# 只运行回忆重构契约测试
pytest tests/unit/test_recall_v1.py tests/unit/test_recall_design_contract.py -q

# 与并列 JShi 仓库的离线接口测试
pytest tests/integration/test_jshi_recall_contract.py -q
```

JShi 双仓测试会在邻接目录不存在 `JShi/src` 时明确跳过。它不会调用 `JSHI_MEMORY_BACKEND=rems3` 启动真实 pipeline，也不会连接模型服务。

## 回忆验收重点

- 主体范围始终受 `subject_id` 限定；对象和时间只在调用方传入时成为硬条件。
- 事件语义向量和对象事实语义向量都只用 L1；词面命中不替代语义信号。
- 结果保留材料类型、对象编号、表示级别和原始检索信号。
- 可及性只能调整相关度接近的结果；强线索仍能命中低可及性材料。
- 总预算选择完整表示，不截断内容；明确请求展开时也必须完整放入预算。
- 命中加强已有记忆；不重写原文、不合成答案、不创建新的记忆单元。
- JShi 当前回忆端口传单个 `object_id`、`level`、片段 `limit` 和锚点；未提供结构化时间范围或字符预算。

详细验收矩阵和待决接口规则见 `design/1610-测试与验收.md`。
