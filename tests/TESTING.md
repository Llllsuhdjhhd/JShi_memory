# REMS 测试体系（2026.06 后）

2026.06 升级后，向量检索改为 **Qdrant 三频段单流**，活跃池改为 **Tier-1 A-Res**，PerfMonitor 双通道隔离。旧测试里大量「双流 Stream A/B + Chroma」假设已失效，测试目录按三层拆分。

## 目录结构

```
tests/
├── conftest.py              # 单元测试：hash 嵌入 + Qdrant :memory:
├── unit/                    # 快测：无 LLM、无 data/，CI 必跑
├── integration/             # 短输入 live LLM（可选 REMS_RUN_LIVE_*）
├── scenarios/               # 长文本场景（红楼梦等），分块 ingest
│   ├── conftest.py          # repo_root、live 门禁、workspace fixture
│   ├── common/              # 场景共用：工作区、LLM 日志、回忆 hook、快照
│   └── hongloumeng/         # 红楼梦：dataset + runner + CLI
├── reports/                 # 人工整理的 live 跑批报告（非自动生成）
└── TESTING.md               # 本文件
```

## 三层职责

| 层级 | 目的 | 依赖 | 运行频率 |
|------|------|------|----------|
| **unit** | 算法与装配正确性 | FakeLLM / hash 嵌入 | 每次 commit |
| **integration** | 真实 LLM + 短句 | `.env` API key | 发版前 / 手动 |
| **scenarios** | 长叙事、跨块状态、诊断 | `data/` + API key | 调参 / 回归 |

## 红楼梦：每次载入一块

数据源：`data/hongloumeng_dataset.json`（约 1300+ 块，每块 300–2000 字）。

### 状态机

- 工作目录：`tests/scenarios/hongloumeng/outputs/<run_name>/`
- `simulation_state.json`：`last_chunk_idx`（已 ingest 的最后一块 id）
- 默认行为：**接着上次继续** ingest 下一块；`--reset` 清空工作区从头来

### CLI（推荐入口）

```powershell
# 仓库根目录
python tests/scenarios/hongloumeng/run_chunk.py              #  ingest 下一块
python tests/scenarios/hongloumeng/run_chunk.py --count 3    #  连续 3 块
python tests/scenarios/hongloumeng/run_chunk.py --chunk-id 5   #  指定第 5 块（仍写 state）
python tests/scenarios/hongloumeng/run_chunk.py --reset --count 1
python tests/scenarios/hongloumeng/run_chunk.py --run-name debug_0429 --offline  # FakeLLM 冒烟
```

### pytest 入口

```powershell
# 数据集结构（无 LLM）
pytest tests/scenarios/hongloumeng/test_dataset.py -q

# 离线一块（FakeLLM + hash，CI 可跑）
pytest tests/scenarios/hongloumeng/test_chunk_runner.py::test_offline_ingest_one_chunk -q

# Live 一块（需 API key + REMS_RUN_LIVE_METABOLISM_TEST=1）
$env:REMS_RUN_LIVE_METABOLISM_TEST = "1"
pytest tests/scenarios/hongloumeng/test_chunk_runner.py -m live_llm -v -s
```

### 每块 ingest 后自动产出

| 文件 | 内容 |
|------|------|
| `chunk_<id>_report.json` | 事件数、角色数、Tier-1 池大小、抽象数、recall_log 行 |
| `chunk_<id>_recall_trace.json` | 三频段检索命中、意图权重、回忆块条目 |
| `llm_calls/chunk_<id>_*.json` | LLM 请求/响应（live 模式） |

## 2026.06 诊断关注点（替代旧双流）

1. **Tri-band**：`intent_weights (α,β,γ)`、各 band 命中、融合 score
2. **Tier-1**：`active_pool` 预过滤前后命中数、`sample_key` 排名
3. **Bypass**：是否触发全库扫描、触发前后 top score
4. **代谢**：Shadow / UC 条数、封存事件数、split_prefix 链
5. **抽象**：`recall_log` 支持度、新抽象 `origin`、ASF 传播（叶子 asf_i）

旧脚本已移至 [`tests/_archive/hongloumeng/`](../_archive/hongloumeng/)。**新跑批请用 `run_chunk.py` + `runner.py`。**

## 环境变量

| 变量 | 含义 |
|------|------|
| `REMS_RUN_LIVE_METABOLISM_TEST=1` | 允许 live_llm 场景测试 |
| `REMS_LIVE_ALLOW_FAKE_EMBEDDING=1` | live 时用 hash 嵌入（省下载模型） |
| `REMS_LLM__API_KEY` | LLM 密钥（`.env`） |

## CI 建议

```yaml
# 必跑
pytest tests/unit/ tests/test_abstraction_service.py -q
pytest tests/scenarios/hongloumeng/test_dataset.py -q
pytest tests/scenarios/hongloumeng/test_chunk_runner.py::test_offline_ingest_one_chunk -q

# 可选 nightly（secrets.REMS_LLM__API_KEY）
pytest tests/scenarios/hongloumeng/test_chunk_runner.py -m live_llm -q
```
