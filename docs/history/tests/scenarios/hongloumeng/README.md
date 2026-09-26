# 红楼梦场景测试（2026.06）

长文本分块 ingest 的推荐入口与目录说明见仓库根 [`tests/TESTING.md`](../../TESTING.md)。

## 嵌入与向量索引

场景跑批默认使用 **语义嵌入**（`embedding.provider=local`，模型 `BAAI/bge-small-zh-v1.5`）。首次在本机运行前需安装 `sentence-transformers` 并允许下载模型缓存。

- **`--offline`**：改用确定性 `hash` 嵌入（无语义，仅冒烟/CI）。
- **变更 act/ent 规则或 embedding provider 后**：必须 **`--reset`** 换新 workspace，或对现有 SQLite 执行 [`scripts/reindex_qdrant.py`](../../../scripts/reindex_qdrant.py) 重建 Qdrant，否则 SQLite 事件与旧向量混存会导致召回失真。

Tri-band 检索默认 **关闭 emo 带向量搜索**（`emo_search_enabled=false`）；情绪仅通过 Mood Modifier 参与重排。

## 回忆 Profile（RagProfile）

场景跑批默认 **`recall_profile=hybrid_literary`**：在 tri-band 多路向量召回之上，增加事件级 BM25（L2 摘要 + 角色名 + 可选 keywords）与近因时间线通道，经多通道 RRF 融合后再做 Factor/Mood 修饰。

- 全局覆盖：`REMS__recall__profile=tri_band` 可回退旧行为。
- 离线对比同一 chunk 的 act_query：`python tests/scenarios/hongloumeng/compare_recall_profiles.py 77`
- `recall_trace.json` 新增 `recall_backend`（profile、channels、channel_sizes）。

- **入库 act**：在 `act_embed_max_chars`（默认 480）内取最长可用摘要（L2→L3… 级联）。
- **查询 act**：pre-recall 对 **search_text（残影+当前块）** 压缩一次（Turn1，**阻塞 recall**）；C/S/E 多路共用。默认 **IngestLlmSession** 三 turn（角色 Turn2 与 recall 并发、不重复贴全文；边界 Turn3 送分句表）；`recall_trace.json` 含 `query_act` 审计。关闭 session：`REMS_INGEST_LLM_SESSION_ENABLED=false`；关闭压缩：`REMS_RECALL_QUERY_COMPRESS_ENABLED=false`。
- 变更入库 act 规则后须 **reindex**。

## 快速开始

```powershell
# 每次 ingest 下一块（默认安静：无 HTTP/Skill INFO、无逐步 Pipeline）
python tests/scenarios/hongloumeng/run_chunk.py

# 需要逐步 Pipeline / HTTP 日志时
python tests/scenarios/hongloumeng/run_chunk.py --verbose

# 连续 3 块
python tests/scenarios/hongloumeng/run_chunk.py --count 3

# 从第 1 块重来（清空 DB + Qdrant，embedding 变更后必做）
python tests/scenarios/hongloumeng/run_chunk.py --reset --chunk-id 1

# 离线冒烟（hash 嵌入，见 test_chunk_runner.py）
python tests/scenarios/hongloumeng/run_chunk.py --offline --count 1

# 仅重建 Qdrant 向量（保留 SQLite 事件）
python scripts/reindex_qdrant.py `
  --db tests/scenarios/hongloumeng/outputs/continuous_run/rems_sim.db `
  --qdrant-path tests/scenarios/hongloumeng/outputs/continuous_run/qdrant_sim
```

## 模块

| 文件 | 职责 |
|------|------|
| `dataset.py` | 读 `data/hongloumeng_dataset.json`，按 id 取块 |
| `runner.py` | 单块/多块 ingest，写 report + recall_trace |
| `run_chunk.py` | CLI |
| `test_chunk_runner.py` | pytest：离线一块 + live 一块 |
| `test_dataset.py` | 数据集格式校验 |
| `compare_recall_profiles.py` | 只读 DB：tri_band vs hybrid_literary 排名对比 |
| `audit_recall.py` | 单块回忆块审计 |

## 旧脚本（已归档）

以下文件在 `tests/_archive/hongloumeng/`，含 2026.05 双流诊断逻辑，与三频段主路径不一致：

- `continuous_simulation.py`、`simulation.py`、`diagnostic_recall.py`
- `ingest_one_chunk_recall_report.py`、`visualize_recall.py`、`phase2_metabolism_diagnostic.py`
- `inspection.py`、`probe_ROLE_SKILL_continuous_db_readonly.py`

新诊断请查看每块产出：

- `outputs/<run>/chunk_<id>_report.json`
- `outputs/<run>/chunk_<id>_recall_trace.json`（intent_weights、tri_band 命中）
