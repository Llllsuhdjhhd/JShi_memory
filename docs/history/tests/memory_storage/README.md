# 记忆存储测试（memory_storage）

针对"记忆写入（ingest）落库"的测试程序。测试输入来自匠石（JShi）与
deepseek 的真实对话：`匠石` 的消息 + `deepseek` 的**最终总结性回复**。

deepseek 轮次规则：

- 只取每一轮对匠石消息的**最终总结性回复**，有长有短都按原文计入，不按字数过滤；
- 判定依据是回复性质（总结性回答），**不是**机械地取文件或对话里的"最后一条"；
- 中间的 commentary 过程消息一律不计入；
- 未完成的轮次（匠石已发、deepseek 总结回复尚未产生）在状态里标注，不计入累计。

## 目录内容

| 文件/目录 | 说明 |
|---|---|
| `run_storage_test.py` | 主程序：包装输入 → ingest → 落库 + 过程痕迹 |
| `conversation.jsonl` | 对话轮次（`speaker` / `content` / `ts`），测试输入源 |
| `memory_storage_test.db` | SQLite 数据库（生成，不入库）：正式表 + `test_*` 痕迹表 |
| `qdrant_test/` | 回忆向量索引（Qdrant 本地模式，生成，不入库） |
| `run_reports/` | 每次运行的 Markdown 报告（生成，不入库） |
| `_selfcheck/` | 离线自检产物（`--fake --force`，不入库） |

## 使用

```bash
# 查看当前累计状态（不建库、不 ingest）
python tests/memory_storage/run_storage_test.py --list

# 追加一轮对话后按阈值判断
python tests/memory_storage/run_storage_test.py --add 匠石 "..." --add deepseek "..."

# 长内容从文件读（UTF-8）
python tests/memory_storage/run_storage_test.py --add-file 匠石 tmp/user.txt

# 强制 ingest + 离线假 LLM（自检）
python tests/memory_storage/run_storage_test.py --force --fake \
    --db _selfcheck/selfcheck.db --qdrant-path _selfcheck/qdrant_selfcheck

# 真实 LLM（需要 API key）
python tests/memory_storage/run_storage_test.py --force \
    --env C:/Users/40575/Desktop/项目/memory/rems_3/.env
```

规则：

- 每轮对话做两件独立的事：①回答你的问题；②机械地继续测试（追加本轮 → 统计
  **待摄入**字数 → 达标 ingest / 未达标等待），直到你明确说停止；
- 已摄入的轮次会标记 `ingested_run=<run_id>`，不会重复摄入；阈值只看**待摄入**轮次
  （默认 1000 字）；
- `--force` 可忽略阈值；
- deepseek 只取每一轮的最终总结性回复（长短不限），中间过程（commentary）不计入。

## 数据库里看什么

正式表（记忆本体）：

- `events`：封存事件（`content_raw` / `summaries` / `role_list` / `emotion` /
  `origin` / `source_ids` / `occurred_at` / `location`）
- `object_memory_entries`：对象时间线（deepseek → OBJ-DEEPSEEK）
- `stored_marks`：`segment_id → 本轮封存事件 id 列表`
- `shadow` / `unclosed_events`：残影与未完成事件缓冲

测试痕迹表（仅测试，`test_*` 前缀，不影响正式 schema）：

- `test_runs`：本次运行汇总
- `test_experiences`：每轮包装后的经历（segment / 说话人 / 原文 / 对象 / 来源）
- `test_llm_calls`：每次模型请求（task_type、模型、延迟、tokens、完整 prompt/response）
- `test_skill_steps`：skill 调用（边界检测 / 事件充实 / 封存）的输入输出摘要
