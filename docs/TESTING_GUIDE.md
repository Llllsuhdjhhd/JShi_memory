# REMS 测试与调优指南

> 适用范围：本指南面向"已经了解白皮书 readme.md 主要术语"的开发者，按 `REMSPipeline.ingest` 的实际执行顺序，逐步说明该测什么、怎么测、看到什么算合格、合格之后还能从哪儿继续调优。
>
> 文档不演练具体测试，只给方案与坐标。需要跑测试时回到 `tests/` 与 `tests/scenarios/`，需要长跑观测时回到 `tests/scenarios/hongloumeng/continuous_simulation.py` 与（待补的）`step_trace.py`。

---

## 0. 一份能跑的环境

### 0.1 Python 与依赖

| 项 | 推荐值 | 备注 |
|---|---|---|
| Python | 3.11.x | 已用 3.11.7 跑通；3.10 也可，3.12 未验证 |
| 依赖管理 | 任意（pip / conda / uv） | 使用 `pyproject.toml` |
| 安装 | `pip install -e .` | 开发模式安装；运行 pytest 必需 |
| 嵌入模型 | `BAAI/bge-small-zh-v1.5` 或 `hash`（占位） | 真模型需要联网 + 第一次下载约 100MB |
| LLM 后端 | OpenAI 兼容协议（DeepSeek / DashScope 等） | `base_url` + `api_key` |

> 推荐第一次先用 `embedding.provider = "hash"` + `FakeLLM` 把全套单测跑通，再上真嵌入与真模型。

### 0.2 关键环境变量

| 变量 | 缺省 | 含义 |
|---|---|---|
| `OPENAI_API_KEY`（或 `REMS_LLM__API_KEY`） | 空 | LLM 凭据；真模型回归必填 |
| `REMS_LLM__BASE_URL` | `https://api.deepseek.com` | 兼容 OpenAI 协议的端点 |
| `REMS_RUN_LIVE_METABOLISM_TEST` | `0` | `=1` 启用 `tests/integration/test_wp_live_*` 的真模型回归 |
| `REMS_LIVE_ALLOW_FAKE_EMBEDDING` | 空 | `=1` 时 wp-live 测试落到 `embedding.provider="hash"`，省下载 |
| `REMS_LOG_DIR` | `./logs/llm_failures` | LLM JSON 解析失败时的样本落盘目录（P3-16 修复后启用） |
| `REMS_WP_REPORT_DIR` | `./reports` | wp-live 自动生成的 markdown 报告输出位置 |
| `REMS_WP_LIVE_NARRATIVE_ROUNDS` | 空 | wp-live 单人剧本只跑前 N 句，便于短程试跑 |

> Pydantic v2 的环境变量解析规则：嵌套字段用双下划线连接，例如 `REMS_LLM__TASK_MODELS__SUMMARY=qwen3-8b`。

### 0.3 一次性自检命令

```bash
python -m compileall -q src tests
python -m pytest tests             # 仅默认套件 + collection 自检
python -m pytest tests/unit -q     # 最快回归，10 秒级
```

健康基线：默认 `pytest tests` 应当 **50 passed + 4 skipped**（4 个 skipped 是 wp-live integration，等手动启用真模型时才跑）。

---

## 1. 测试分层

```
┌───────────────────────────────────────────────────────────────┐
│ Layer 1：单元 (tests/unit/*)                                   │
│   - 纯纯函数 / 单 service 行为；FakeLLM；in-memory Chroma；秒级 │
├───────────────────────────────────────────────────────────────┤
│ Layer 2：白皮书对齐 (tests/test_*.py)                          │
│   - prompt 形态、签名兼容、跨模块约定（ingest 主流程子段断言）  │
├───────────────────────────────────────────────────────────────┤
│ Layer 3：场景脚本 (tests/scenarios/hongloumeng/*)              │
│   - 真嵌入 + 真 LLM 长跑；产出 markdown 报告                   │
│   ① continuous_simulation.py：批量灌入 → 累积曲线 / 分布      │
│   ② step_trace.py（待补）：单 chunk 深追踪每步中间产物         │
├───────────────────────────────────────────────────────────────┤
│ Layer 4：wp-live 集成 (tests/integration/test_wp_live_*.py)    │
│   - 真模型 + InstrumentationStore/WPReporter；只在显式开启时跑 │
└───────────────────────────────────────────────────────────────┘
```

**建议节奏**：

1. 改完代码先跑 Layer 1 + Layer 2（默认 `pytest tests`，必须全绿）；
2. 改了 prompt 或评分策略，再跑 Layer 3 单 chunk 深追踪（看每个中间产物是否符合预期）；
3. 全套调通后做一次 Layer 3 长跑（看累积现象：抽象触发频率、未完成库膨胀、遗忘衰减分布）；
4. 上线前做一次 Layer 4 wp-live 全套，看白皮书条款的真模型可观测断言。

---

## 2. 按 Pipeline 流程的功能测试

主流程在 `src/rems/pipeline.py::REMSPipeline.ingest`，下面 8 个 step 对应代码上看得见的阶段。**每一步**都按"观测点 / 评价标准 / 测法 / 调优旋钮 / 已知坑位"五要素展开。

### Step 0｜输入接收 + 残影读取

**做了什么** — `meta_repo.get_shadow()` 从 `MetabolismRepository` 取出当前残影；它**严格等于**所有未完成事件 `merged_content` 的拼接（白皮书 §4.1）。

**观测点**

- `Shadow.content`、`Shadow.length`；
- `MetabolismRepository.get_unclosed_events()` 当前条目数与各自 `merged_content` 长度。

**评价**

| 指标 | 期望 |
|---|---|
| `len(Shadow.content)` | ≤ `physical_redline`（10/66 × context_chars） |
| Shadow 与 UC 拼接是否一致 | `\"\\n\".join(ue.merged_content for ue in unclosed) == Shadow.content` |
| UC 数量随时间分布 | 不应单调增长——若是，多半是 P0-5 修复未生效 |

**测法**

- Layer 1：`tests/unit/test_metabolism_service.py`；
- Layer 3 单 chunk：`step_trace.py` 在 ingest **前后**各 dump 一次 `(shadow.length, len(unclosed))`，断言"unclosed 总长 == shadow 长度"。

**调优旋钮**

- `physical_redline / safe_watermark`：来自 `context_window` × `chars_per_token` 的派生值；
- `unclosed_force_ratio`（默认 1.2）：单条 UC 强制封存的红线倍率。

**已知坑**

- 若把 `chars_per_token` 设小（< 1.0），`len_msg` 会被压到 < 1000 字，所有真实输入都触发兜底封存。
- shadow / UC 不一致通常是因为有人绕过 `_apply_boundary_result` 直接调 `delete_unclosed_event`——别这么做。

---

### Step 1｜Pre-recall 角色抽取（确定焦点角色）

**做了什么** — 用 (shadow + raw_input) 跑一次 `RoleExtractionSkill.extract`，得到本轮所有可能涉及的角色，去重 + register；同时拿出"焦点角色 ID 集合"喂给召回，并把 `Role` 列表当作 `known_roles_hint` 一路带到 `seal_event`，让 `EventEnrichmentSkill` 拿去做代词消解（P1-6 修复）。

**观测点**

- LLM `task_type=\"role_extraction\"` 的 user 消息内容（角色提取 prompt）；
- 返回的 `ExtractedRole.role_id / name / importance / snapshot / emotion`；
- `RoleService.resolve_and_register` 的别名/新建逻辑：是否复用已有 role、是否产生重复别名；
- `focus_role_ids` 集合是否包含 `core_user_role_id`（单人模式）。

**评价**

| 指标 | 期望 |
|---|---|
| 主体角色（"我" / "张三"）应被识别为 S 级 | 单人模式下 `core_user_role_id` 永远在 `focus_role_ids` |
| L1/L2/L3 snapshot 字段按重要性填写 | S 必有 L1/L2/L3，A 只有 L1/L2，B/C/D 只有 L1 |
| 代词消解 | 多人模式下"他/她"应被映射到 `active_participants` 之一，不应回落到核心用户 |
| 同一角色多次出现 | `role_repo.get(role_id)` 命中已有 ID，不创建重复 |

**测法**

- Layer 1：mock LLM 返回固定 JSON，断言 `role_repo.list_all()` 数量；
- Layer 3 单 chunk：`step_trace.py` 在该步 dump 抽到的 `[(name, importance, role_id)]` 列表与 `focus_role_ids`；
- Layer 4：`tests/integration/metrics/probes.prompt_contains_user_mode_block` 验证 prompt 注入了模式块。

**调优旋钮**

- `user_mode` / `core_user_role_id` / `active_participants`：白皮书 §2.2 模式块；
- `task_models.role_extraction`：把这一步换更便宜模型可以显著降本（同时调 `event_enrichment`）；
- prompt：`src/rems/llm/prompts.py::ROLE_EXTRACTION_SYSTEM` + 模式块；改"S/A/B/C/D 的快照层级分配策略"是最常修的位置。

**已知坑**

- 第一次跑会产生若干"幻影"角色（环境/物体被识别为 person）；解决方案是在 system prompt 加"必须是叙事中具有意向性的实体"或在 `RoleService.resolve_and_register` 加白名单过滤。
- pre-recall 的角色集合不一定与每条 sealed event 的真实角色子集一致；当前架构把它当 hint 而不是硬覆盖（P1-6 修复点）——若你看到"不该出现的角色被绑到了某条事件"，**不要**把 hint 改成硬覆盖，反而该收紧 enrichment prompt。

---

### Step 2｜双流回忆（Stream A 语义 + Stream B 白描，70/30 + RRF）

**做了什么** — `RecallService.build_recall_block(...)`：

1. **Stream A**：基于 query 在事件向量库做语义检索；**未达 `recall_max_capacity` 走全局**，超过则 70% 最近段全局可见、30% 旧段仅"焦点角色白名单标志位 (`role_<id>: True`) "命中（P0-3 修复）。
2. **Stream B**：基于 query + 焦点角色 snapshot 文本在 white-painting 集合检索；用 `effective_distance = distance / max(effective_forgetting, ε)` 让遗忘因子直接作用于排位（P1-7 修复）。
3. **RRF + Modifier**：每流按"距离升序"独立得 Rank，`Final = (1/(k+R_A) + 1/(k+R_B)) * Factor_Modifier(只对流B) * Mood_Modifier(同号情绪共振)`。
4. **role_aware tier 选择 + redline 截断**：按角色重要性选 L1/L2/...；累计长度逼近 `physical_redline * recall_intermediate_filter_factor` 时停止追加。

**观测点**

- `vector_store.count()` 是否触发 70/30；
- 两流的 hit 数（`InstrumentationStore.vector_calls`）；
- 每条命中的 `(distance, rank_a, rank_b, factor_modifier, mood_modifier, final)`；
- `RecallBlock.items[*].summary_level` 与 `len(item.content)`；
- `recall_block.total_length`。

**评价**

| 指标 | 期望 |
|---|---|
| 库容量未到上限 | 流 A 走全局；70/30 不启用 |
| 焦点角色对长尾老事件的"反向激活"路径有效 | 30% 段命中应当带 `role_<id>` 等值匹配；空 focus 时不可用 |
| 两流并集 vs 单流 | 焦点角色相关时 RRF 比单流高（典型测法见 `tests/unit/test_recall_service.py::test_rrf_merge_uses_both_streams`） |
| `recall_block.total_length` | ≤ `physical_redline * 1.2` |
| 头部条目 | 应保留更精细档（L1/L2），尾部允许压到 L3+/Ultra |

**测法**

- Layer 1：`tests/unit/test_recall_service.py`（已覆盖 RRF + 时间衰减）；
- Layer 2：`tests/test_dynamic_recall_v2.py`（头/尾压缩 + redline 守卫）；
- Layer 4：`probes.recall_block_budget` + `probes.recall_has_ultra_fallback` + `format_hybrid_line` 把每命中的分项分数写进报告。

**调优旋钮**

- `recall_max_capacity / recall_global_ratio`：什么时候启用 70/30、最近段比例；
- `recall_rrf_k / recall_factor_alpha / recall_mood_beta`：RRF + Modifier 的形状；
- `recall_default_tier_offset / recall_primary_role_detail_shift / recall_minor_role_compress_shift`：tier 选择；
- `recall_intermediate_filter_factor`：截断红线相对 `physical_redline` 的倍率；
- `wp_half_life_days / forgetting_silence_threshold`：遗忘节奏（间接影响 stream B 排名）；
- 白皮书 §4.4 改造后**不再**有"绝对量混合"——若想引回相似度阈值过滤，请在 `_get_stream_a_hits` 里加 `distance` 上限剪枝，**不要**重新引入 cosine 加权融合。

**已知坑**

- 70/30 依赖 Chroma metadata 里有 `create_time` 与 `role_<id>: True` 标志位（P0-2/P0-3 修复点）；老库需要重新 reindex 才能用上分层。
- `_hybrid_score` 已经是 *legacy hook*（仅供 `install_recall_scoring_tracer` 诊断使用），不要再当成主排序函数。
- Stream B 在没有焦点角色时**完全为空**，是设计意图；这种场景下 RRF 退化成"流 A 单流"，不是 bug。

---

### Step 3｜recall_log 登记

**做了什么** — 把本次 recall block 中真实事件 ID（去重，含基本与抽象）追加到 `recall_log_repo`。这是**唯一**驱动抽象事件诞生的输入流（白皮书 §3.2）。

**观测点**

- `recall_log_repo.list_all()` 增长曲线；
- 每行 `event_ids` 长度分布。

**评价**

| 指标 | 期望 |
|---|---|
| 静默倾听模式 | 即便外部不返回 ContextPackage，本轮也必须登记一条 recall_log（否则永远不会演化出抽象） |
| 同一事件 ID 跨条目复现率 | 反映"长期共现"信号；过低 → 抽象永远不触发，过高 → 抽象过早合成（细节流失） |
| 抽象合成后的 ID 替换 | `replace_subset` 调用应把回忆历史里的子集换成抽象 ID，长尾继续在同一命名空间挖掘 |

**测法**

- Layer 1：`tests/test_abstraction_service.py`（重写后的版本，覆盖了 below_support / meets_support / idempotent 三种路径）；
- Layer 3 长跑：`continuous_simulation.py` 的累积面板里画出"recall_log 行数 / 抽象触发次数"双曲线。

**调优旋钮**

- `abstract_subset_min_size`（默认 6）/ `abstract_subset_min_support`（默认 5）：小数据集联调时改成 3/3 比较容易看到抽象出现，生产环境保持默认。

**已知坑**

- "静默倾听"模式（如 `ProcessingMode.PASSIVE_LOG`）必须仍然完整执行回忆 + 登记；如果你新增模式时只在 `returns_context_package=False` 上做手脚，极易把抽象演化整路堵死。

---

### Step 4｜代谢（边界检测 + 残影/未完成库维护）

**做了什么** — `MetabolismService.process_input(...)`：

1. 拿当前 UC 与拼接后的 shadow；
2. `BoundaryDetectionSkill.detect(shadow_content, current_input, unclosed)`：
   - 分别对 shadow / current 做分句，prompt 中显式标注两段区间（P1-9 修复）；
   - LLM 返回 `completed_events[].content_raw_indices` + `new_unclosed_indices`；
3. `_apply_boundary_result`：
   - `continuation_of` 命中的旧 UC 被消费删除；
   - **未被命中的旧 UC 全部清空**（白皮书 §4.2.2 trace decay；P0-5 修复）；
   - 用 `new_unclosed` 重建未完成库，残影 = 重建后 UC 的拼接；
   - 单条 UC 长度越过 `len_msg × unclosed_force_ratio` 强制封存为可疑事件。

**观测点**

- 边界 prompt：是否带"既有残影 / 本轮新输入"区间提示；
- LLM 返回的 `completed_events` 与 `new_unclosed_indices`；
- 本轮的 `(consumed_uc_ids, new_uc_ids, force_sealed)` 三元组；
- ingest 完成时 `(shadow.length, sum(len(uc) for uc in unclosed))`。

**评价**

| 指标 | 期望 |
|---|---|
| 防碎片化 | 同一段落里若干琐碎动作合并为单事件；不应每句一条 |
| 续写判定 | "继续上次的话题"应正确填 `continuation_of=UC-xxx` |
| 残影/UC 不变量 | `Shadow.content == "\\n".join(uc.merged_content)`，且**长跑时长度有界** |
| 强制封存 | 单条 UC 长度逼近 `len_msg * 1.2` 时应当被 force-seal，不应永远挂着 |

**测法**

- Layer 1：`tests/unit/test_metabolism_service.py`；
- Layer 2：`tests/test_whitepaper_alignment.py::test_boundary_prompt_marks_shadow_and_current_ranges`；
- Layer 3 单 chunk：`step_trace.py` 在每轮 dump LLM 返回的 indices 与 force-seal 决策；
- Layer 3 长跑：`continuous_simulation.py` 的"unclosed_total_length(t)" 曲线必须是带状有界（不是 monotonic 增长）。

**调优旋钮**

- `len_msg`（间接通过 `context_window / chars_per_token` 调）；
- `unclosed_force_ratio`：保守取 1.2，激进取 1.5（少强制兜底）；
- prompt：`BOUNDARY_SYSTEM` / `BOUNDARY_USER`；最常调的位置是"防碎片化"与"残影中应继续挂起的句子也要重新出现在 `new_unclosed_indices`"两段。

**已知坑**

- LLM 漏报"残影中应继续挂起"的句子时，按设计直接 trace decay；如果业务不能容忍丢内容，**应在 prompt 强化提示**而不是在代码里"每条旧 UC 都强制保留"——后者会把 P0-5 修复反向推回。
- 真模型对 indices 编号偶尔会越界；`_decode_new_unclosed_list` 已对扁平/嵌套两种格式做兜底。

---

### Step 5｜事件封存（Enrichment + Decoration + 索引）

**做了什么** — `EventService.seal_event(content_raw, ..., known_roles=...)`：

1. 计算 `CompressionBudget`（白皮书 §1.2 前置预算）；
2. 单次 `EventEnrichmentSkill.enrich(content_raw, known_roles=hint, skip_roles=?)`：
   - 默认 `full_mode`：一次 LLM 同时拿到 `summaries{L1..Ln}` 与 `roles[]`（P1-6 修复）；
   - 外部已给 `role_entries` / `skip_roles=True` 时：走 summary-only 分支；
3. `RoleService.resolve_and_register` 把 enrichment 抽出的角色映射到稳定 ID；
4. 可选 `_generate_decoration`（`enable_decoration=False` 时不调 LLM）；
5. 计算 `compression_ratio = sum_len / raw_len`；
6. EMA 演化（`EMAEvolver.evolve_event`）+ 计算 `activation_energy`；
7. 落盘 + 写 Chroma metadata（`create_time` + `role_<id>: True` + `is_abstract` + `event_length` + ...）。

**观测点**

- enrichment 的 user 消息（应当包含 `已知角色列表 / 摘要字预算 / 角色快照预算表`）；
- 返回的 `summaries` / `roles` 字段；
- `event.compression_ratio`（应接近 `compression_target_ratio = 0.1515` ± 容忍带）；
- Chroma metadata 实际写入字段（`vector_store._memory_events[event_id]['metadata']`）；
- EMA 后的 `event.role_list[*].emotional_model.valence/arousal`、`event.event_valence`、`event.activation_energy`。

**评价**

| 指标 | 期望 |
|---|---|
| 摘要熔断 | 某级 `len(summary) ≤ summary_fuse_min_chars` 后**不再**生成更高级（白皮书 1.1.3） |
| L1 字数 | ≈ `raw_len × compression_target_ratio × budget_ratio_summary` |
| 角色 snapshot 层级分配 | S 必须 L1+L2+L3；A 必须 L1+L2 |
| 索引 metadata | 必须含 `create_time`（POSIX 秒）+ 每个 role 的 `role_<id>: True` |
| `compression_ratio` | 落在 `[0.1, 0.3]` 区间（短文本会因宽恕机制偏高，正常） |

**测法**

- Layer 1：`tests/unit/test_event_service.py`；
- Layer 2：`tests/test_whitepaper_alignment.py::test_event_service_seal_event_basic`；
- Layer 3 单 chunk：`step_trace.py` 在该步 dump `(prompt_len, summaries.keys(), len(roles), compression_ratio, metadata.keys())`；
- Layer 4：`probes.white_painting_tier_check` 验证白描档位与 cfg 一致。

**调优旋钮**

- `compression_target_ratio` / `compression_budget_multiplier`：整体松紧；
- `summary_decay_factor` / `snapshot_decay_factor`：层级衰减节奏；
- `budget_ratio_*`：`summary/snapshot/wp/decoration` 四项总和应=1.0；
- `enable_decoration`：默认关，开启后每条事件多 1 次 LLM 调用，谨慎；
- `task_models.event_enrichment`：成本最敏感的旋钮——这一调用包含摘要 + 角色，能换更弱模型直接砍 50% 成本。

**已知坑**

- enrichment full prompt 比 summary-only 长～30%，但**省了一次 role_extraction 调用**——总 token 比旧实现少。
- `known_roles` 是 hint **不是**硬覆盖；若强行用它替代 LLM 抽取，跨事件角色子集就会失真。
- 旧路径 `pre_summaries` 已删；如果你看到 KeyError/TypeError，就说明有旧测试在传 deprecated kw。

---

### Step 6｜角色更新（白描时间线 + 语义卡片 + 遗忘因子）

**做了什么** — `RoleService.update_from_event(event)`：

1. 对每个非抽象事件的角色追加白描条目（`l3_decision / l2_interaction / l1_mention` 之一，按 `wp_*_field` 选档）；
2. 必要时刷新角色 *Semantic Card*（聚合白描产出的"概念词云 / 性格摘要"）；
3. 维护 `forgetting_factor` —— 新条目重置为 100，旧条目按 `wp_half_life_days` 自然衰减；
4. 抽象事件直接跳过这一步（白皮书 §3.1 抽象事件 `role_list=[]`）。

**观测点**

- `RoleRepository.get_white_painting(role_id, limit=...)` 的尾部（最近 N 条）；
- 单角色总字符容量与 `wp_role_capacity` 的比值；
- 各角色的 `forgetting_factor` 分布；
- 语义卡片刷新触发条件（默认 N 条新白描或时间窗触发；详见 `RoleService` 实现）。

**评价**

| 指标 | 期望 |
|---|---|
| 白描连续性 | 同一角色的 wp 时间线随事件单调追加 |
| 容量内不惩罚 | `total_chars < wp_role_capacity` 时，`forgetting_factor` 不衰减 |
| 容量外软遗忘 | 超过容量后老条目 `forgetting_factor` 在召回打分中被使用 |
| 抽象事件 | 不应往任何角色 wp 里追加条目 |

**测法**

- Layer 1：`tests/unit/test_role_service.py`（含 abstract_events_skipped）；
- Layer 3 长跑：跑 100+ 条输入，画"每个核心角色 wp 累计字数 / 遗忘因子分布"曲线。

**调优旋钮**

- `wp_role_capacity_divisor`（默认 6.6）：单角色容量；
- `wp_half_life_days`（默认 60）：自然衰减节奏；
- `event_silence_threshold` / `forgetting_silence_threshold`：低于阈值的事件 / 白描条目被视作"已遗忘"；
- `wp_primary_field / wp_default_field / wp_minor_field`：白描收集端档位（白皮书 §2.3 动态粒度路由）。

**已知坑**

- 长跑时若不开启遗忘衰减（`wp_half_life_days` 设很大），主要角色的 wp 会变成"日历"——召回会一直把它打到顶；这不是 bug，是参数问题。
- 语义卡片刷新如果走的是 LLM，注意 `task_models.summary` 的成本，必要时改为"按周期"而非"按条"。

---

### Step 7｜抽象合成（频繁子集挖掘）

**做了什么** — `AbstractionService.mine_and_synthesize()`（白皮书 §3.2 唯一触发路径）：

1. 拉所有 recall_log，把每行 event_ids 视作一条交易；
2. 用 closure-style 频繁子集挖掘（`_find_all_frequent_subsets`），筛 size ≥ `abstract_subset_min_size` 且 support ≥ `abstract_subset_min_support` 的全部频繁子集（非仅极大）；
3. 跳过已经合成过的（`AbstractedSubsetRepository.is_fired`）；
4. 每个候选子集：
   - `evidence_policy.collect`（默认 `LeafContentRawEvidencePolicy`：递归展开抽象链到叶子，拼接 `content_raw` 做证据）；
   - `InductiveEvolutionSkill.synthesize` 单次 LLM 产出 `content_raw + (optional) insight`；
   - **复用 EventEnrichmentSkill**（P1-8 修复）一次调用产出全部层级摘要，**不再循环 N 次**；
   - 落盘 + 索引（同样写 `create_time`）；
   - `recall_log_repo.replace_subset`：把回忆历史里该子集换成抽象 ID，方便高阶抽象继续挖。

**观测点**

- `recall_log_repo.list_all()` 行数；
- 每次 `mine_and_synthesize` 返回的新抽象事件数；
- 抽象事件的 `source_events` 与 `abstraction_level`；
- `AbstractedSubsetRepository.list_fired()` 的累积；
- 抽象事件的 `summaries{L1..Ln}` 字数曲线。

**评价**

| 指标 | 期望 |
|---|---|
| 子集大小 vs support | 单次回忆永远不触发；多次回忆重叠才触发 |
| 抽象层级递增 | 高阶抽象的 source_events 可以包含低阶抽象事件，`abstraction_level` 单调 +1 |
| 摘要 LLM 次数 | 抽象事件总 LLM 次数 = 1 (synthesize) + 1 (enrichment) = **2 次**；旧实现是 1 + N |
| recall_log 替换 | 抽象后 `replace_subset` 应改写多条回忆行 |

**测法**

- Layer 1：`tests/test_abstraction_service.py`（覆盖 below_support / meets_support / idempotent 三路径）；
- Layer 3 长跑：`continuous_simulation.py` 跑 200+ 条输入，画"抽象事件累计 vs 时间"折线。

**调优旋钮**

- `abstract_subset_min_size / abstract_subset_min_support`：触发频率；
- `enable_abstract_insight`：默认 False，关闭时不要求模型产 insight；
- `evidence_policy`：默认 `LeafContentRawEvidencePolicy`；想做"主题词压缩证据"或"基于角色 snapshot"的证据，**新增策略**而不是改默认；
- `task_models.abstraction / task_models.insight`：可独立切到更强的模型（成本敏感）。

**已知坑**

- 不要回退到旧的"向量近邻聚类触发"思路：抽象的**唯一**输入是 recall_log。
- 抽象事件的 `role_list = []`、不写白描、不写语义卡片；如果你看到抽象事件出现在白描时间线里，是 Step 6 的回归 bug。

---

### Step 8｜ContextPackage 输出（按 ProcessingMode 分流）

**做了什么** — 主干所有内部记忆动力学跑完后，根据 `ProcessingMode.returns_context_package` 决定外部交付形态：

| Mode | 外部返回 | 备注 |
|---|---|---|
| `DIALOGUE`（默认） | `ContextPackage = recall_block + shadow + current_input` | 给上游 LLM 生成回复 |
| `PASSIVE_LOG`（静默倾听） | `None` | 内部回忆 / 抽象仍正常执行 |
| `NPC_AGENT` | `ContextPackage` + `npc_directives` | 多轨指令（行动 / 对话 / 内心独白） |

**观测点**

- 返回值是否符合该模式的 shape；
- `npc_directives` 在 NPC 模式下不为空。

**评价 / 测法**

- Layer 4：`probes.dialogue_package_shape / passive_log_silence / npc_directives_shape` 一并验证；
- Layer 1：`tests/unit/test_pipeline.py::TestPipelineConstruction` 已经验证模式枚举与 `ingest` 不同 mode 的返回类型。

**调优旋钮**

- 新增模式只需要在 `ProcessingMode` 添成员 + 重写 `returns_context_package` 与（可选）专属装配；不需要动主干。

**已知坑**

- 误把"静默倾听"做成"短路 ingest"会切断 recall_log 登记——重述一遍：**所有模式都必须执行 Step 2/3/4/5/6/7**，模式只决定 Step 8 怎么对外。

---

## 3. 命令速查

```bash
# Layer 1：默认套件（FakeLLM + in-memory Chroma）
python -m pytest tests/unit -q
python -m pytest tests/test_*.py

# Layer 2：单文件聚焦
python -m pytest tests/test_whitepaper_alignment.py -k boundary_prompt -v

# Layer 3a：单 chunk 深追踪（待实现 step_trace.py）
python tests/scenarios/hongloumeng/step_trace.py --chunk-id 12 --dump-dir reports/step_trace_12

# Layer 3b：长跑累积面板
python tests/scenarios/hongloumeng/continuous_simulation.py --rounds 200 \
    --report-dir reports/sim_run_$(date +%Y%m%d_%H%M)

# Layer 4：wp-live 真模型回归（带报告）
$env:REMS_RUN_LIVE_METABOLISM_TEST = "1"
$env:REMS_LIVE_ALLOW_FAKE_EMBEDDING = "1"        # 想用真嵌入则去掉
$env:REMS_WP_REPORT_DIR = "reports"
python -m pytest tests/integration/test_wp_live_single_mode_narrative.py -v
```

---

## 4. 调优 Roadmap（按性价比）

| 等级 | 调优点 | 见效场景 | 风险 |
|---|---|---|---|
| ⭐⭐⭐ | 把 `task_models.event_enrichment` / `role_extraction` 切到便宜模型 | 立刻削 50% LLM 成本 | 角色识别质量下降 → 用 `tests/scenarios/hongloumeng/step_trace.py` 验收 |
| ⭐⭐⭐ | 调 prompt：`BOUNDARY_SYSTEM` 防碎片化 + `ENRICHMENT_FULL_SYSTEM` 角色去代词化 | 直接影响事件粒度与召回质量 | 改完务必重跑 Layer 2/3 |
| ⭐⭐⭐ | `abstract_subset_min_size / min_support` 与场景规模匹配 | 抽象触发频率 | 默认 6/5 偏保守，小数据集（<50 chunk）适当下调 |
| ⭐⭐ | 召回打分：`recall_factor_alpha / recall_mood_beta` | 流 B 命中权重 / 情绪共振强度 | 改大易"主角戏过强" |
| ⭐⭐ | 70/30：`recall_max_capacity / recall_global_ratio` | 大库性能与召回准确度 | 库 < 1000 时无差别 |
| ⭐⭐ | 新增 `AbstractionEvidencePolicy`：例如"角色 snapshot 证据" | 抽象事件主旨更清晰 | 别改 default，新增策略并通过 DI 注入 |
| ⭐ | EMA：`emotion_decay_lambda_per_hour / ema_history_window` | 情绪平滑节奏 | 影响 `event_valence`，间接影响 mood_modifier |
| ⭐ | 白描档位：`wp_primary_field / wp_default_field / wp_minor_field` | 收集端粒度 | 与 Step 5 budget 联动 |
| ⭐ | EnrichmentResult 解析失败兜底：`role_fallback` | 真模型 JSON 失稳时的健壮性 | 当前已注入；若想关闭，构造 `EventEnrichmentSkill(..., role_fallback=None)` |

**潜在但还未实现的优化方向**（非本次清单内）：

1. **embedding 入库批量化** — `event_service._index_event` 现在每条事件单独 upsert；批量处理可显著降低 ChromaDB write IO。
2. **recall_log 大表的频繁子集挖掘**优化 — `_find_all_frequent_subsets` 是 O(n²·|avg|)，实际数据 < ~1k recalls 还够用；规模上去之后建议引入 FP-Growth 或 LCM。
3. **白描容量超限后的"软遗忘 + 主动凋亡"二阶段** — 当前只在 Recall 打分中用遗忘因子；真正彻底淘汰旧条目需要在 `RoleService` 层加周期性 garbage collection。
4. **角色合并 / 别名归并** — 长跑会出现"张三 / 三哥 / 老张"分别存为 3 个 role_id；可以加一个角色合并 service 用 LLM 仲裁 + 更新 white_painting 与 event 的 role_id。
5. **InsightService** — 当 `enable_abstract_insight=True` 时，把 insight 单独抽出来做"跨抽象事件元规律"的二阶聚合（白皮书 §3.3）。
6. **`step_trace.py`** — 深追踪脚本目前是 TODO；建议结构：`--chunk-id` 单条 ingest → 每步前后 dump `(prompt, llm_response, entities_changed)` 三元组到 markdown，作为白皮书条款的"逐句答辩"入口。

---

## 5. 故障排查清单

| 症状 | 大概率根因 | 验证方法 | 修复位置 |
|---|---|---|---|
| `pytest` 报 `cannot import name 'ProbeResult'` | 老的 `tests/integration/metrics/` 被删 / 修改 | `python -c "from tests.integration import metrics; print(metrics.ProbeResult)"` | `tests/integration/metrics/__init__.py` 末尾 `from . import probes` 必须放在 `ProbeResult` 定义之后 |
| 长跑 `unclosed_total_length` 单调增长 | P0-5 未生效 / 被改回 | 单跑 10 轮 ingest 看 `len(repo.get_unclosed_events())` 是否回零 | `metabolism_service._apply_boundary_result` 里的"清空未命中旧 UC"段 |
| 70/30 没启用 | metadata 缺 `create_time` | 读 `vector_store._memory_events[<eid>]['metadata']` | `event_service._index_event` / `abstraction_service._index_abstract` |
| 多人模式代词总落到核心用户 | system prompt 模式块没注入 | dump 任一 LLM 调用的 system message | `prompts.build_user_mode_block` 是否被各 `build_*_system_message` 覆盖 |
| LLM JSON 解析失败但找不到样本 | 旧版本写到 cwd 的 `failed_llm_json.txt` | 现在写到 `$REMS_LOG_DIR/failed_llm_json_<UTC>_<rand>.txt`（P3-16） | `src/rems/llm/provider.py::_write_failed_llm_json` |
| 抽象事件永远不出现 | `min_size / min_support` 太大或 recall_log 行数不够 | `recall_log_repo.list_all()` 长度 vs `min_support` | 调小阈值 / 增加输入轮数 |
| 召回主角"戏太重" | `recall_factor_alpha` 偏大 | 看 `format_hybrid_line` 输出的 factor_modifier 列 | 调小 `recall_factor_alpha` 至 0.3 |
| compression_ratio > 1.0 | 短文本 + 宽恕机制 + multi 角色 | 检查 `event.event_length < 200` 时是否正常 | 期望行为；不需要修。如果长文本仍然 > 0.5，说明 prompt 让模型抄原文，要收紧 enrichment 的"L1 字预算" |

---

## 6. 维护这份文档

每次出现下述变化时，请同步更新本文件：

- 新增 / 删除一个 `ProcessingMode`；
- 新增 / 删除一个 `Skill`（`src/rems/skills/*`）；
- 在 `REMSConfig` 里加 / 改 / 删配置项（特别是带"白皮书 §x.y"注释的那种）；
- `_apply_boundary_result` / `build_recall_block` / `mine_and_synthesize` 的不变量改变。

文档里的"已知坑位"是踩过的雷，删除请慎重——保留下一位维护者的成本远低于看错文档的成本。
