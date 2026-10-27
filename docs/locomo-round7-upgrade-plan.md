# 第七轮：分阶段记忆升级的 LoCoMo 验证方案

日期：2026-10-24
状态：方案已定，尚未执行付费运行
关联提交：`11d9cf9`（100–500 字轻量事件、L1 代表句、上层记忆附带原文）
相关文档：[分阶段提炼](multistage-memory-refinement.md)、[LoCoMo 评测](locomo-evaluation.md)、[回忆](recall.md)

## 结论

现有运行器可以继续用，但不能直接复用第六轮的 `--small-events-only` 方案。

第六轮 `--small-events-only` 是“取消全部到龄聚合”的对照：从旧库恢复阶段 0 小事件，跳过所有提炼，
也不会有 L1 和附加原文。它恰好绕开这次升级最需要验证的部分。直接用这个方案测，只能证明阶段 0
写入没有退化，不能回答“提炼后是否还会丢细节”。

这次应新写入一个空库，跑完整的 `staged` 生命周期，再从同一个新库做一次 small-only 对照：

- 7A：新代码、新空库、默认 90/365 天提炼，验证升级后的端到端行为。
- 7B：复制 7A 的阶段 0 事件，关闭提炼，作为同一批事件的对照。
- 两者共用同一条历史、同一 30 题、同一评分条目和同一 JShi 文本入口。

## 这次升级要验证什么

| 升级点 | 需要看到的证据 | 不能只靠总分判断 |
|---|---|---|
| 小事件只在 100–500 字内封存 | 阶段 0 长度分布、超限兜底数量、未封存残余 | 事件变短不等于信息更完整 |
| 到龄后才提炼 | 90 天阶段的触发数量、隐藏来源数、每个父事件的来源链 | 只跑阶段 0 不能验证提炼 |
| L1 为每个直接子事件写代表句 | `〔#序号〕` 覆盖全部子事件、`代表性溢出` 有记录 | 摘要分数高不等于代表句在 |
| 检索命中上层后附第一层原文 | `detail_attached` 次数、实际交付材料中的 `原文 #`、预算拒绝原因 | source_id 命中不等于核心看到原文 |
| 一年阶段不再继续提炼 | 时间模拟回归；LoCoMo 本身不触发 | 不能把 166 天历史说成多年验证 |

## 为什么第六轮方案不够

第六轮的两个目录 `runs/locomo_round6_v2_small_only/` 和 `runs/locomo_round6_v3_small_only/`
都明确记录了 `additional_age_refinement: false`，并把 `visible_events` 恢复为全部阶段 0 事件。
它们的结论是“不聚合时成绩更好”，不是“升级后的提炼已经可用”。

另外，本轮升级改变了事件形成，不能拿第六轮的封存库当单变量基线：

- 第六轮小事件上限是 2000 字，实际最大 1484 字；本轮检查线是 500 字。
- 第六轮没有 L1 代表句，也没有 `detail_attached`。
- 本轮的审计产物和 `retrieval_evidence` 都新增了字段；旧运行缺少这些字段。

所以 7A 必须重新写入。7B 可以复用 7A，因为 7B 要对照的正是“同一批新小事件，提炼开/关”。

## 数据与时间边界

固定第一组 `conv-26`，与第五、六轮相同：

| 项目 | 数值 |
|---|---:|
| 会话数 | 19 |
| 原始发言 | 419 |
| 原始字符数 | 66,318 |
| 历史区间 | 2023-05-08 至 2023-10-22 |
| 跨度 | 166 天 |
| 评测时钟 | 2023-10-23 |
| 90 天截止 | 2023-07-25 |
| 到龄会话 | 1–10（10 个会话，约一半发言） |
| 未到龄会话 | 11–19（9 个会话） |

默认 30 题（seed=42）按证据所在会话拆分：

- 21 题的证据全部落在 1–10 会话，通常会经过 L1 提炼。
- 8 题的证据落在 11–19 会话，仍读取阶段 0 原文。
- 1 题跨前后两段。

跨会话事件按最新来源日期判断是否到龄，靠近截止线的题可能仍保持阶段 0；
分组统计时以实际 `refinement_hidden_by` 为准，不只看证据所在会话号。

这个分布适合本轮：总分主要由“经过提炼的早期问题”决定，同时保留 8 题近期原文作为同轮对照。
但报告不能只给一个总分，必须把“到龄/未到龄”分开看。

一年阶段（365 天）在这条 166 天历史上不会触发。它只能由 `tests/unit/test_memory_refinement.py`
的时间模拟覆盖，不能拿本轮 LoCoMo 成绩声称验证了多年交互。

## 推荐运行设计

### 7A：完整 staged（升级主验证）

- 输出：`runs/locomo_round7_staged/`
- 新空库，完整回放 419 条发言
- `memory_refinement.enabled=true`，阶段 90/365 天
- 默认 JShi 投递条件、英文整词召回、L0 索引
- 30 题，seed=42，固定评分条目
- 预算：`--budget 30`（见下面的累计账本说明）

### 7B：同一批小事件，关闭提炼（对照）

- 输出：`runs/locomo_round7_small_only/`
- `--reuse-baseline runs/locomo_round7_staged`
- `--small-events-only`：只把 7A 的阶段 0 事件复制到新的对照库，重置可见性；
  不重新提取、不重新回放历史，也不改动 7A 的源库
- 其余参数与 7A 相同

7A 与 7B 的差异只剩下“是否把到龄事件提炼成上层事件，以及是否附带原文”。
事件切分、原始正文、来源、题目和评分都相同。

### 可选 7C：扩大题量

如果 7A/7B 的差异落在少数题上，单轮 30 题不足以判断稳定性。可以另开目录把题量扩到
50–60 题，或用全部 199 题；不要在原目录追加，因为 manifest 已冻结题集。
扩题主要增加回答和评分费用，不需要重新写入历史。

不建议一开始就做旧的 `888e463` 代码对照：它能分离“升级前/后”，但要另建 worktree、
另跑一次完整写入，费用和变量都更多。先用 7A/7B 回答“升级后的提炼是否可用”。

## 执行命令

以下命令在仓库根目录、已激活 `jshi-memory-test` 的终端执行。本机当前解释器是
`C:\Users\Lenovo\miniconda3\envs\jshi-memory-test\python.exe`。

### 7A：分阶段执行，先检查再回答

```powershell
python -B scripts/locomo_jshi_eval.py --phase prepare `
  --out runs/locomo_round7_staged --questions 30 --seed 42 `
  --default-memory-triggers --english-word-lexical `
  --event-index-level L0 --memory-lifecycle staged --budget 30

python -B scripts/locomo_jshi_eval.py --phase ingest `
  --out runs/locomo_round7_staged --questions 30 --seed 42 `
  --default-memory-triggers --english-word-lexical `
  --event-index-level L0 --memory-lifecycle staged --budget 30
```

`ingest` 阶段会完成历史回放和到龄提炼。先不要继续回答，检查：

```powershell
Get-Content runs/locomo_round7_staged/baselines/c0/complete.json -Raw
Get-Content runs/locomo_round7_staged/history/c0/refinement.jsonl
```

如果 `refinement.jsonl` 出现 `errors`，或 `deferred` 发生在很靠前的位置，先停下来分析；
不要带着“只提炼了几条”的库继续跑 30 题，否则结果会被误读成升级失败。

如果 `ingest` 中途失败，已写入的 `rems.db` 没有 `complete.json`，同一目录会被拒绝重复回放；
按现有规则应换新目录，不能把不完整库用于评分。

确认后继续：

```powershell
python -B scripts/locomo_jshi_eval.py --phase answer --out runs/locomo_round7_staged --questions 30 --seed 42 --default-memory-triggers --english-word-lexical --event-index-level L0 --memory-lifecycle staged --budget 30
python -B scripts/locomo_jshi_eval.py --phase score  --out runs/locomo_round7_staged --questions 30 --seed 42 --default-memory-triggers --english-word-lexical --event-index-level L0 --memory-lifecycle staged --budget 30
python -B scripts/locomo_jshi_eval.py --phase report --out runs/locomo_round7_staged
```

### 7B：复用 7A，关闭提炼

```powershell
python -B scripts/locomo_jshi_eval.py --phase run `
  --out runs/locomo_round7_small_only --questions 30 --seed 42 `
  --default-memory-triggers --english-word-lexical `
  --event-index-level L0 --memory-lifecycle staged `
  --small-events-only --reuse-baseline runs/locomo_round7_staged `
  --budget 30
```

7B 会复制 7A 的 `rubrics/`，不会重新调用评分条目生成；如果 7A 的 manifest 还没冻结题集，
先完成 7A 的 `prepare`。

### 对照报告

```powershell
python -B scripts/compare_locomo_rounds.py `
  --first runs/locomo_round7_staged `
  --second runs/locomo_round7_small_only
```

它会在 7B 目录写 `comparison.md/json`，并检查两轮的 dataset/JShi/seed/questions/rubrics 是否配对。
7B 的 provenance 标记 `view=small_events_only`，比较结果会记为
`same_small_events_refinement_toggled=true`：同一批阶段 0 事件，只切换提炼，不是重新提取，
也不是“同一封存库仅重建索引”。

## 运行前检查

1. **预算账本是跨轮累计的。** `runs/locomo_budget.jsonl` 当前已结算
   `¥12.741908`，未结算 `0`。`BudgetLedger` 用累计值检查 `--budget`，所以
   `--budget 10` 会在第一次付费调用时被拒绝；本轮用 `--budget 30`。
   如果 7A 花费超过约 `¥17.26`，7B 会被同一个上限挡住；按历史单轮费用，这种情况不应发生。
2. **不要在 7A 和 7B 之间修改 `../JShi/src/jshi`。** manifest 会冻结
   `jshi_code_sha256`；当前 JShi 工作区有语音和提示词样式的未提交改动，
   `memory/subject/skill/identity/recognition` 路径本身没有 diff，但哈希仍会变。
   先固定当前状态，跑完两轮再继续改 JShi；若要和旧轮严格比较，先记录这些差异。
3. **不要复用第六轮 baseline。** 事件长度和提炼行为都变了；`--reuse-baseline` 还会因
   `jshi_code_sha256`/questions 不匹配而拒绝。
4. **确认英文编码器可用。** 本机已有 `BAAI/bge-small-en-v1.5` 缓存；`ingest`/`answer`
   会用 `local_files_only=True` 加载。可选先跑免费检查
   `python scripts/prepare_english_eval.py`，它只下载/验证编码器，不调用付费 API。
5. **确认 `runs/locomo_round7_staged` 不存在或为空。** 运行器会复用已有 manifest；
   设置不同会被拒绝，未完成的 baseline 也会阻止重复写入。为避免混用，用新目录。

## 运行后验收

### 1. 写入与提炼结构

从 `complete.json` 和 `refinement.jsonl` 记录：

- `events`、`small_events`、`small_event_max_chars`
- `visible_events`、`hidden_events`、`refinement_as_of`
- `refinement` 每轮 `refined`/`deferred`/`errors` 数量
- 阶段 0 事件数、阶段 1 父事件数
- 每个父事件的 `source_events`、`L1` 长度、`representative_overflow`
- 419 个原始来源是否全部被覆盖

结构底线：

- 阶段 0 不生成摘要、情绪或角色分析。
- 父事件正文等于其子事件正文（阶段 1）或子事件 L1（阶段 2）的逐字拼接。
- 每个被吸收的子事件都有指回父事件的隐藏链接。
- L1 中能看到每个直接子事件的 `〔#序号〕` 代表句。
- 普通回忆不单独返回隐藏节点。

### 2. 附加原文是否真的发生

`report.md` 的“召回证据诊断”只统计后端 `retrieved_events`，不直接统计附加原文。
需要从 `answers/*.json` 的 `recall_candidates` 单独数：

- `reason="detail_attached"`：真正附上的第一层原文
- `reason="detail_below_threshold"`：相似度不够
- `reason="detail_no_room"`：预算放不下
- `reason="detail_limit"`：被 `max_count=1` 挡住

再看 `delivered_memory` 是否包含 `原文 #` 和对应正文。只有“候选里有”不算交付；
只有 `source_id` 命中也不算核心看到。

### 3. 问答结果

`summary.json` 至少记录：

- 语义正确率、充分信息覆盖率、完整覆盖题目比例、官方 F1
- 分题型结果
- 到龄题（21 道）与未到龄题（8 道）的分组结果
- 7A 与 7B 的逐题对照，不只比较总数

本轮是同一批开发题的单轮对照，不是独立留出集，也不是显著性检验。模型回答和裁判都有随机性；
总分小幅变化不能单独归因于提炼。要结合逐题材料区分：

- 没召回
- 召回了没交付
- 交付了没答对
- 答对了但裁判判错

### 4. 成本与完整性

检查 `summary.json.budget`、`runs/locomo_budget.jsonl` 和 `api_audit.jsonl`：

- 未结算预留应为 0
- 没有 `usage_unknown`
- 总费用在预期范围内

## 已知测量缺口

这些不影响“能不能跑”，但会影响“结果怎么解释”，本轮记录时要保留：

1. **`audit_refinement.py` 会对新策略误报。** 它把“阶段 0 超过 `small_event_max_chars`”和
   “L1 超过三分之一预算”直接记为 violation；新策略允许“切不开整段保存”和
   “代表句放不进预算时保留并记 `representative_overflow`”。跑完 7A 后要按
   `refinement_policy.representative_overflow` 和 `seal_reason` 人工区分，不能只看脚本退出码。
2. **`retrieved_events` 不包含附加原文的子事件 ID。** 轨迹条目只有父事件；
   `detail_event_ids` 没有写进 `RecallTraceItem`。因此报告里的证据诊断会低估“只通过附加原文交付”的覆盖。
   本轮用 `recall_candidates` 和 `delivered_memory` 补看；若要把这项变成正式指标，应在
   `answer_question` 里把 `detail_attached` 的子事件并入诊断，或给轨迹条目加字段。
3. **`refine_history` 遇到 `deferred` 会停止。** 它只统计 `status="refined"` 的批次；
   如果模型没有分出任何组，循环就结束，`complete.json` 仍会写。7A 的 `refinement.jsonl`
   必须单独检查 `deferred` 的位置和 `selected_ids`；必要时先修提示/解析再重跑，不能把半提炼库当完整结果。
4. **文档里“末尾残余不封存”的说法需要核对。** `runtime.finalize_history` 调用
   `process_input("", force_save=True)`，代码路径会走 `_force_save_all` 并封存残余；
   第五轮 `complete.json` 的 `residual_events_closed` 是 0（当时没有残余）。
   7A 若有残余，先看这个字段和 `unclosed_events` 表，再决定是否能说“只能靠词面找到”。
5. **`--budget 10` 是旧的首轮默认/示例，不是当前可用的值。** 账本已经累计 `¥12.741908`；
   新轮必须传更大的累计上限，本文用 30。

## 成本与时间预估

历史单轮参考：

- 第五轮 v3 从空库完整写入 + 30 题，比 v2 增加约 `¥1.26`
- 第六轮 small-only 单轮增加约 `¥0.47`
- 30 题回答本身约 65 秒；主要时间在历史写入和提炼调用

7A 预计 `¥1.5–3`、20–40 分钟；7B 预计 `¥0.5–1`、10–20 分钟。
总计约 `¥2–4`，在 `--budget 30` 内。实际以 `api_audit.jsonl` 的 usage 为准，不把估算当账单。

## 本轮不验证的内容

- 后台常驻调度、真实时间流逝和失败重试；本轮用回放时钟同步调用。
- 超过 1000 事件的大库概率采样、长期遗忘和持续学习。
- 365 天一年阶段的真实问答；只用回归/时间模拟。
- 附加原文的完整展开（默认最多 1 条）；不是“把所有子事件都给核心”。
- 生产 JShi 的语音/JEV 决策；本轮仍是文本入口、观察者视角。

## 最小结论模板

跑完后至少回答这四个问题，再写总结：

1. 7A 是否完成了预期的 90 天提炼，还是中途 `deferred`/只提炼了少数事件？
2. 阶段 0 的长度、来源和隐藏链接是否守住结构底线？
3. 附加原文实际发生了多少次，是否进入核心交付材料，而不是只在候选里？
4. 7A 相对 7B，在到龄题上恢复/损失了哪些具体事实？未到龄题是否基本不变？
