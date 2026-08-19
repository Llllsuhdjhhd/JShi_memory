# REMS 工作指南

本文面向后续参与 REMS 的开发者。目标是让项目在主体框架稳定的前提下，持续改进提示词、遗忘机制、RAG 召回、抽象策略和模型路由，而不是反复改动主流程。

## 1. 核心原则

REMS 的主干流程应保持稳定：

1. 全局人物前置提取（仅辅助定位焦点角色，服务于召回）。
2. 构建回忆上下文（包含双流交集召回与遗忘因子过滤）。
3. 登记 `recall_log`（触发记忆恢复与加强）。
4. 代谢输入，边界切分并生成基本事件。
5. 事件级人物精确提取，并更新角色白描与语义卡片。
6. 基于 `recall_log` 挖掘频繁子集，生成抽象事件。
7. 根据模式决定是否对外返回上下文。

后续改进尽量落在策略接口、Prompt 模块、模型配置和评测集上。除非白皮书架构本身变化，不要随意改 `REMSPipeline.ingest` 的主干顺序。

## 2. 目前稳定的模块边界

- `EventService`：基本事件封存、**事件级精确角色提取**、摘要、装饰、索引。
- `RoleService`：角色注册、白描时间线（**含动态遗忘衰减与静默控制**）、语义卡片。
- `RecallService`：**双流召回（语义与人物交集）**、混合打分、**记忆加强恢复**、摘要档位选择、回忆块组装。
- `AbstractionService`：频繁子集挖掘、抽象事件合成、抽象索引。
- `MetabolismService`：残影、未完成事件、边界切分、事件封存调度。
- `InductiveEvolutionSkill`：抽象事件 LLM 合成。
- `EventEnrichmentSkill`：基本事件摘要与角色精确抽取。
- `RoleExtractionSkill`：人物快照提取（用于管道前置全局扫描与事件后置精确提取）。
- `BoundaryDetectionSkill`：事件边界切分。
- `VectorStore`：向量索引与检索。
- `REMSConfig`：系统级开关、预算、模型映射和算法参数。

## 3. 推荐改动位置

### 3.1 Prompt 持续改进

优先改这些位置：

- `src/rems/llm/prompts.py`
- `src/rems/llm/prompt_registry.py`
- 各 `Skill` 中 prompt 组装逻辑

建议做法：

- 不要直接把 prompt 越写越长。每次改动应对应一个明确坏例。
- 新 prompt 应有版本意识，例如 `event_enrichment:v1`、`event_enrichment:v2`。
- Prompt 改动必须配套至少一个回归样例。
- 重点关注：
  - 边界切分是否过碎或漏切；
  - 角色是否漏抽、误合并；
  - L1-L10 是否保持递归压缩；
  - 抽象事件是否使用基本事件 `content_raw`，且保留关键角色；
  - insight 是否为提炼，不是事实复述。

### 3.2 遗忘机制改进

优先改策略接口：

- `src/rems/strategies/forgetting.py`

**当前已落地的代谢与遗忘算法**：
- **初始权重**：由情感能量（AE）决定极值；
- **自然代谢**：基于时间半衰的指数衰减；
- **静默过滤**：遗忘因子低于 0.02 物理级不可见；
- **回忆恢复**：成功进入回忆块获 1.5 倍增益。

后续可以继续替换或扩展：

- FIFO 容量溢出时的极速惩罚策略；
- 角色级长期偏好或特定白描类型的抗遗忘权重；
- 人工置顶或绝对冻结记忆（永不衰退）；
- 特定场景的动态衰减系数调节。

默认策略应继续保持可解释、可测试。

### 3.3 RAG 与召回改进

优先改策略接口：

- `src/rems/strategies/recall.py`
- `src/rems/services/recall_service.py`
- `src/rems/storage/vector_store.py`

**当前核心架构（双流记忆召回）**：
- **流 A**：语义相关性查询。
- **流 B**：基于焦点人物的反向查询（附加动态遗忘惩罚）。
- **合并策略**：交集优先算法（强语义 + 强人物锚点），结合余量回填。

后续可改进点：

- embedding 模型与多路混合；
- 交集优先算法中的动态阈值调整；
- 引入轻量级 rerank 模型精排；
- 摘要档位选择依据动态预算的进一步平滑；
- 彻底超预算时的极端 fallback 策略。

原则：`RecallService` 负责流程编排，具体打分和档位判断应放到策略类里。

### 3.4 抽象机制改进

优先改：

- `src/rems/strategies/abstraction.py`
- `src/rems/services/abstraction_service.py`
- `src/rems/skills/inductive_evolution.py`

当前稳定规则：

- 抽象触发来自 `recall_log` 的频繁子集（全部满足阈值的项集，非仅极大）。
- 抽象合成输入应展开到叶子基本事件。
- 使用基本事件 `content_raw` 与角色线索作为证据。
- 抽象事件不登记结构化角色。
- `insight` 由 `enable_abstract_insight` 控制。

后续可扩展：

- 频繁子集挖掘算法；
- 证据选择策略；
- 抽象事件长度预算；
- 抽象 insight 策略；
- 高阶抽象去重；
- 抽象质量评估。

### 3.5 模型路由改进

优先改：

- `src/rems/config.py`
- `src/rems/llm/provider.py`

当前通过 `TaskModelMapping` 按任务选择模型。后续建议支持：

- 每类任务独立模型；
- fallback 模型；
- prompt 版本与模型版本绑定；
- 任务级 temperature；
- 任务级 max tokens；
- 本地模型和远程模型切换；
- 失败重试与降级策略。

### 3.6 角色消解改进

优先关注：

- `RoleService.resolve_and_register`
- `RoleExtractionSkill`
- `EventEnrichmentSkill`

**当前已落地的两阶段提取架构**：
1. **全局前置扫描**：对残影+输入做一次人物扫描，仅为 Recall 提供焦点角色（不传给事件）。
2. **事件级精确提取**：对切分干净的事件内容再做一次人物快照，确保不发生动作混淆。

可扩展方向：

- 更强的实体消歧；
- 别名合并；
- 可疑角色确认；
- 角色合并 / 拆分工具；
- 单人模式与多人模式的独立测试集。

角色 ID 稳定性会强烈影响长期记忆质量，相关改动必须谨慎。

### 3.7 摘要与压缩预算改进

优先关注：

- `EventService._compute_budget`
- `SummaryGenerationSkill`
- `EventEnrichmentSkill`
- `REMSConfig` 中的压缩参数

可扩展方向：

- L1-L10 衰减曲线；
- 短文本宽恕机制；
- 角色快照预算；
- decoration 预算；
- 抽象事件特殊预算；
- 熔断规则。

## 4. 评测优先级

后续最重要的不是盲目调 prompt，而是建立黄金评测集。

建议至少覆盖：

- 单人生活日志；
- 多人对话；
- 长篇叙事文本；
- 模糊代词；
- 多角色交互；
- 未完成事件；
- 抽象事件触发；
- 抽象事件再抽象；
- 角色长期状态变化；
- 墓碑与冲突修正；
- RAG 召回排序。

每个样例最好标注：

- 应切分出的事件数量；
- 每个事件的 `content_raw` 主干；
- 应出现的角色；
- 不应出现的角色；
- L1 必须保留的信息点；
- 是否应触发抽象；
- 抽象事件应保留的角色主干；
- insight 是否应生成。

## 5. 开发流程

推荐流程：

1. 先写或选择一个失败样例。
2. 确认失败属于 prompt、算法、模型还是数据问题。
3. 在对应策略或 Skill 中做最小改动。
4. 跑定向测试。
5. 再跑相关回归测试。
6. 更新工作指南或白皮书中对应规则。

不要在没有样例的情况下大幅改 prompt 或算法。

## 6. 测试环境

本项目推荐使用 conda 环境：

```powershell
conda run -n py3125 python -m pytest <test_path> -q
```

常用定向测试：

```powershell
conda run -n py3125 python -m pytest tests/unit/test_inductive_evolution.py -q
conda run -n py3125 python -m pytest tests/unit/test_strategy_interfaces.py -q
conda run -n py3125 python -m pytest tests/unit/test_recall_service.py::TestScoring -q
```

红楼梦场景测试位于 `tests/scenarios/hongloumeng/`，用于检查长篇叙事文本下的事件切分、角色抽取、摘要、白描与 LLM 调用日志。

先校验数据集：

```powershell
conda run -n py3125 python -m pytest tests/scenarios/hongloumeng/test_dataset.py -q
```

运行红楼梦仿真：

```powershell
conda run -n py3125 python tests/scenarios/hongloumeng/simulation.py
```

仿真会读取 `data/hongloumeng_dataset.json`，默认处理若干 chunk，使用 hash embedding，并调用真实 LLM。输出目录在：

```text
tests/scenarios/hongloumeng/outputs/sim_runs/
```

检查最近一次仿真结果：

```powershell
conda run -n py3125 python tests/scenarios/hongloumeng/inspection.py
```

红楼梦专项观察重点：

- 是否把章回叙事切成适度大小的基本事件，而不是过碎或整章吞入；
- `content_raw` 是否保留原文主线和关键角色；
- 角色列表是否覆盖专名、神怪、叙述主体等可区分实体；
- L1 摘要是否保留人物、动作、因果与场景；
- 白描时间线是否能形成角色连续状态；
- 抽象事件是否基于基本事件 `content_raw`，且不漏关键角色；
- LLM 调用日志中的 prompt 和 response 是否保持中文可读，不出现大量转义或结构错误。

编译检查：

```powershell
conda run -n py3125 python -m compileall src/rems
```

## 7. 提交规范

提交前确认：

- 没有把无关文件放进同一个 commit；
- 没有提交 `.env`、密钥、数据库、临时日志；
- Prompt 改动有对应测试或样例说明；
- 策略接口改动保持默认行为兼容；
- 重要架构规则同步更新 `readme.md` 或本指南。

建议提交粒度：

- prompt 改进单独提交；
- 策略接口单独提交；
- 算法替换单独提交；
- 测试样例可以与对应修复一起提交；
- 文档同步可以跟随架构改动提交。

## 8. 当前架构判断

当前 REMS 的主体框架已经可以作为稳定主干继续推进。后续主要工作不应是重写主流程，而是围绕以下层面持续迭代：

- Prompt 版本化；
- 策略接口；
- 黄金评测集；
- 模型路由；
- 可观察日志；
- RAG 与遗忘算法；
- 角色稳定性。

如果需要大改主流程，应先更新白皮书规则，再给出迁移计划和回归测试范围。
