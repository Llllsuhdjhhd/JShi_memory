# WP-LIVE 白皮书指标报告

本目录为 **Live 测试生成的 Markdown 报告**默认输出位置。历史快照已移至 [`../_archive/reports/`](../_archive/reports/)。

## 运行前提

- 配置 `.env` 中的 `REMS_LLM__API_KEY`
- `REMS_RUN_LIVE_METABOLISM_TEST=1`

## 命令（使用 integration 套件）

```powershell
python -m pytest tests/integration/test_wp_live_single_mode_narrative.py -v -s
python -m pytest tests/integration/test_wp_live_multi_and_modes.py -v -s
python -m pytest tests/integration/test_wp_live_abstraction_and_recall.py -v -s
python -m pytest tests/integration/test_metabolism_live_first_20_inputs.py -v -s
```

可选：`REMS_WP_REPORT_DIR` 指定报告输出目录（默认 `tests/reports/`）。

## 嵌入

Live 测试默认使用 SentenceTransformer；无法加载 PyTorch 时可设 `REMS_LIVE_ALLOW_FAKE_EMBEDDING=1`（hash 向量，语义质量不保证）。
