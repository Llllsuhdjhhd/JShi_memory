# 历史测试与脚本归档（2026.06 前）

本目录存放**不再纳入默认 pytest 收集**的旧测试、诊断脚本与报告快照。  
2026.06 升级（Qdrant 三频段、Tier-1、单流回忆）后，其中大量内容仍假设 **Chroma + 双流 A/B**，仅作参考保留。

## 目录

| 子目录 | 内容 | 替代方案 |
|--------|------|----------|
| `root_live/` | 根目录重复的 `test_wp_live_*`、`test_metabolism_live_*` | `tests/integration/test_wp_live_*.py` |
| `metrics/pkg/` | 旧 `tests/metrics/`（与 integration 重复） | `tests/integration/metrics/` |
| `hongloumeng/` | 2026.05 红楼梦长跑/诊断脚本 | `tests/scenarios/hongloumeng/run_chunk.py` + `runner.py` |
| `reports/` | 某次 Live 跑批生成的 Markdown 快照 | 重新跑 integration 测试生成新报告 |

## 勿直接 pytest 本目录

`pyproject.toml` 中已设置 `norecursedirs` 排除 `_archive`。

若需手动运行归档脚本（自担兼容性风险）：

```powershell
python tests/_archive/hongloumeng/continuous_simulation.py 1
```
