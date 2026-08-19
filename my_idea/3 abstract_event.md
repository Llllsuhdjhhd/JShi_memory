## 3. 抽象事件定义与防幻觉规范

抽象事件（Abstract Event）通过对多次回忆中反复共现的低阶事件子集做归纳，形成具备跨时间属性的高阶认知节点。

### 3.1 核心属性解析

结构与基本事件一致，但存在以下硬性差异：

- **content_raw**：大模型基于 source_events 展开得到的叶子基本事件 content_raw 拼接输入，按基本事件同样的"抓住重点"规则压缩生成；目标长度约等于叶子基本事件 content_raw 均值的 1.2 倍；
- **summaries**：按 §1.1.3 递归摘要体系基于 content_raw 正常生成（L1…Ln + 弹性浮动与 <40字符截止），与基本事件同规则；
- **insight**：由 `enable_abstract_insight` 开关控制。开启时由 **InductiveEvolutionSkill 在同一次 `synthesize()` 调用中**与 `content_raw` 一并产出（不另挂 skill、不二次 LLM 调用）：须显式携带语义后验仲裁通过的**五大认知关系常量**之一（如 `CAUSALITY`），以及对应的自然语言认知结论——前者标注事件以何种逻辑被归纳，后者阐述归纳出的规律，均不得复述 `content_raw` 中的原始事实；关闭时不生成；
- **is_abstract**：固定为 True；
- **role_list**：**恒为空**。合成输入必须带入基本事件的角色线索，压缩 `content_raw` 时不能漏掉关键角色/实体；当需要完整角色信息时，通过 `source_events` BFS 展开至叶子基本事件获取（见 §1.1.8）。

- **abstraction_level**：记录在递归演化树中的深度，支持抽象事件再作为其他抽象事件的子集成员参与更高阶合成。

### 3.2 抽象事件的触发机制（唯一路径：频繁子集挖掘）

抽象事件的合成不再依赖任何向量近邻聚类或定期扫描。其唯一触发路径为：回忆块事件集合上的频繁子集挖掘。

- **登记阶段**：每次 RecallService.build_recall_block 成功产出回忆块后，将块内所有真实事件 ID（即 basic 与 abstract，过滤 CARD:* 等伪条目）的有序去重集合，作为一条记录写入 recall_log 表。

**【修订】挖掘阶段**：对 recall_log 的历史记录族 $T_1, T_2, \dots$（每个 $T_i$ 为一次回忆的事件 ID 集合），采用**闭频繁项集挖掘**替代普通频繁项集挖掘。

#### 3.2.1 闭频繁项集挖掘（2026.06 新增）

系统**仅挖掘闭频繁项集（Closed Itemsets）**——即不存在任何超集具有相同支持度的频繁项集。这从数学上保证了提取的是"最大化、最完整的记忆共现基元"，避免了普通频繁项集挖掘产生的指数级冗余子集。

- **算法选型**：后台采用内存高效的 FP-Growth 变体（如 FP-Close）或基于垂直数据格式的 Eclat 算法定期离线扫描。
- **挖掘窗口**：系统仅在最近 K 次回忆（`abstract_mining_recent_k`，参数可配置，默认 1000）内执行挖掘。$K$ 直接决定搜索空间大小，是控制后台算力消耗的核心阀门，受 §4.6 双通道独立耗时监控的后台负载因子动态调节。

#### 3.2.2 时间衰减支持度（2026.06 新增）

支持度计算不再使用简单计数，而引入指数时间衰减，使得近期共现的模式获得更高权重：

$$\text{Support}(S) = \sum_{T_i \supset S} e^{-\lambda(t_{now} - t_i)}$$

其中 $\lambda$ 为时间衰减系数（可配置）。该机制防止陈年旧事因累计基数大而无限阻碍新认知规律的生成。

保留所有同时满足下述条件的闭频繁子集 $S$：

- $|S| \geq abstractsubsetminsize$（默认 6，可配置）；
- $\mathrm{Support}(S) \geq abstractsubsetminsupport$（默认 12，可配置）。

候选按 $|S|$ 与支持度排序，单轮内倾向先处理较大子集，以减轻与 replace_subset 的交互扭结。

#### 3.2.3 语义后验与五大认知关系约束（2026.06 新增）

统计学上的频繁共现并不等同于逻辑相关。对每个通过前述支持度筛选的闭频繁项集 $S$，依次经过两道 LLM 关卡：

**关卡一 · 叙事整合性前置闸门（`abstract_narrative_coherence_enabled`，默认开启）**：在合成前，先用一次 `InductiveEvolutionSkill.evaluate_narrative_coherence()` 把挖矿子集划分为"相干叙事线"与"无关项"。其相干占比即白皮书 §3.2 所述的**相关率 $a$**，会回灌 `RecallQualityController` 的 EMA-$a$ 通道（参与动态语义距离钳制）。

- 若相干事件数 $<$ `abstract_narrative_coherence_min_count`（默认 3），整组判为伪共现，标记 `coherence_rejected` 跳过本次抽象，不持久化；
- 否则仅保留相干子集进入合成，丢弃无关项，避免统计噪声污染抽象证据；
- 关闭该开关时跳过此关卡，全量子集直送合成（仍受关卡二的 NONE 仲裁约束）。

**关卡二 · 语义后验合成**：对（过闸后的）相干子集，**调用一次** `InductiveEvolutionSkill.synthesize()`：在同一次 LLM 请求中完成认知关系仲裁、压缩合成 `content_raw` 与（开关开启时的）`insight` 产出——**insight 与认知关系仲裁不再单独挂接额外 skill 或发起额外 LLM 调用**（即合成本身仍是单次请求；整合性闸门是它之前的一道独立前置调用）。

LLM 须先仲裁该子集的认知拓扑关系；若判定为"纯属巧合的随机共现 (NONE)"，则**直接抛弃**，不持久化抽象事件。仲裁通过时，同一次响应一并输出 `content_raw` 与 `insight`（`enable_abstract_insight` 开启时）。`insight` 须同时包含**关系类型标识**（下列五大常量之一）与**自然语言认知结论**——前者标明"这些事件以何种逻辑被归纳"，后者阐述"归纳出了什么规律"，二者均不得复述 `content_raw` 中的原始事实。

合法的五大认知关系常量（写入 `insight` 的关系类型标识）：

1. **CAUSALITY**（因果链条）：前置事件是后置事件的起因或动机（如：遭暗算 → 负伤 → 寻医）。
2. **PROTOTYPE_INVARIANT**（原型共性/类比）：跨时空、跨客体的相似遭遇闭环（如：反复遭遇同一类商业套路）。
3. **TEMPORAL_CHRONO**（习得性时序/常规）：高频习惯性行为序列（如：例会与周报）。
4. **COGNITIVE_DIALECTIC**（认知冲突与转折）：信念的打破与重建（如：极度信任 → 遭遇背叛）。
5. **MERONYMY**（图式成员/部分整体）：微观动作向宏观状态的归属（如：扫地、看小说归属于居家休闲）。

- **合成阶段**：对每个闭频繁子集 $S$，以其成员作为 `source_events`；**单次**调用 `InductiveEvolutionSkill.synthesize()`——输入为成员递归展开后的叶子基本事件 `content_raw` 与角色线索，输出 `content_raw` 及（开关开启时的）含关系类型标识的 `insight`；持久化后**另**调用 §1.1.3 的 `SummaryGenerationSkill`，**仅**为 `content_raw` 生成多级摘要与 `actual_max_level`，再索引入向量库。（摘要与 insight 分属不同 skill，但 insight 不脱离 InductiveEvolutionSkill 独立生成。）
- **幂等 & 同名空间迭代（抽象事件可再次抽象）**：
  - 每条抽象事件生成后会把其指纹（$S$ 成员排序后拼接）登记到 abstracted_subsets 表，同一 $S$ 不会被重复合成；
  - 生成后会遍历所有满足 $S \subseteq T_i$ 的历史 recall_log 行，将其中的 $S$ 成员移除，插入新抽象事件 ID。"用抽象事件 ID 代替原来的子集"使得挖掘逻辑在单一命名空间内统一：后续回忆块若仍然整体命中该抽象事件，会与历史记录在同一空间里累积支持度；若多个已合成的抽象事件再度频繁共现，即触发更高阶抽象（source_events 可直接嵌套抽象事件 ID，abstraction_level 递增），自然形成多阶演化树；
  - source_events 内的基本事件会被标记 is_abstracted = True（仅基本事件；嵌套的抽象事件不做此标记，因为它们本身也是高阶演化的中间节点）；
  - **从抽象反查基本事件**：需要溯源到事实层时，调用 EventRepository.resolve_basic_event_ids(event_id) 即可沿整条 source_events 链路 BFS 展开到叶子基本事件，无论抽象有多少层（见 §1.1.8）。这使得"规律 → 事实"的检索在任意阶抽象上都保持 O(1) API。

#### 3.2.4 抽象事件叙事线判重（2026.05 新增，可选）

`mine_and_synthesize` 内部按以下顺序处理：

1. `**is_fired(subset)`**：字面完全相同的子集直接拦截（旧机制）。
2. `**enable_narrative_dedup` 为真时** `**_is_narrative_duplicate(subset)`**：拦截「字面不重复、但 `replace_subset` 也够不着」的灰区——subset 与某条已有抽象事件 $A$ 的**叶子**集合（含嵌套抽象的 BFS 展开）在叙事线层面高度重合：
  - $\mathrm{overlap} = |S \cap A.\mathrm{leaves}| / |S| \geq$ `narrative_dup_overlap_threshold`（默认 0.9）；
  - 同时新增信息**不显著**：$\mathrm{noveltyratio} = |S \setminus A.\mathrm{leaves}| / |S| <$ `narrative_dup_novelty_min_ratio`（默认 0.1） **或** $\mathrm{noveltyabs} <$ `narrative_dup_novelty_min_abs`（默认 2）；
  - 任一条件成立则跳过新合成，并把这条 subset 以 `matched_id`（被命中的 A）登记到 `fired_repo`，避免下一轮反复挖到再走一次判重。
2.5. **叙事整合性前置闸门**（`abstract_narrative_coherence_enabled`，**默认开启**，见 §3.2.3 关卡一）：`_synthesize_from_subset` 内先做一次 LLM 划分（相干叙事线 vs 无关项）。相干数 $<$ `abstract_narrative_coherence_min_count` 则 `coherence_rejected` 跳过；否则只用相干子集进入合成。相干占比 rate_a 回灌 `RecallQualityController` 的 EMA-$a$。
3. **合成成功后的 `replace_subset(...)`**：把所有"完全包含 $S$"的 recall_log 行原子级改写——把 $S$ 的成员替换为新抽象 ID；多叙事线共享同一叶子时，仅**完全包含** $S$ 的行被改写（见 §3.2 幂等段）。

**默认**：`enable_narrative_dedup=False`，不跑第 2 步；`is_fired` 与 `replace_subset` 仍为互补的硬收敛手段。

**护栏意义**（判重开启时）：单独看 overlap 比例会被"短子集差 1 个"骗过（5/6=0.83 ≈ 边界），单独看绝对量会让"长 subset 加 1 个新事件"被误放过；**显著新增**须 novelty 比例与绝对量**同时**达到阈值才视作"老素材新叙事"（白皮书 §3 关于"同一基础事件可在多条叙事线上独立抽象"的语义）。

**性能护栏**（判重开启时）：候选 × 已有抽象的全比成本随抽象规模线性增长。为保障检索效率，判重比对窗口被严格限制在最近 K = `narrative_dup_compare_recent_k`（默认 200）条已合成抽象。若遭遇极端高并发，应用层可依据 §4.6 的抽象通道监控，降级切至极小比对窗口（如默认 20），避免过度消耗后台算力。

### 3.3 幻觉控制与防强化循环

- **危险的自我强化循环规避**：如果系统仅用于后台记录（无需通过记忆进行预测或对话输出），则不存在该问题。
- 若应用于多轮交互场景，为了防止抽象推演结论被反复当作原始事实入库造成幻觉闭环，归纳演化技能在合成时始终展开到叶子基本事件，并以基本事件 content_raw（而非中间级摘要或上一轮抽象文本）作为证据输入。抽象事件参与更高阶合成时同样先反查到基本事实层。

### 3.4 抽象遮蔽因子与信息守恒算法（2026.06 新增）

为解决多阶抽象生成后，底层基本事件与高阶摘要同时被召回导致的"认知复读"问题（同一段事实被反复呈现），系统引入**抽象遮蔽因子（Abstraction Shadowing Factor, ASF）**，通过数学机制动态调节被抽象囊括的底层事件的检索可见性。

#### 3.4.1 单跳 ASF 估算算法（无需 LLM 算力）

基本事件 $A$ 被抽象事件 $B$ 囊括后，计算 $B$ 对 $A$ 的直接遮蔽量。复用已有的向量与长度特征进行低成本估算：

$$ASF_{A \to B} = \max\left(0, \ \cos(V_A, V_B) \times \min\left(1.0, \frac{\text{Length}_B}{\text{Length}_A}\right)\right)$$

- **逻辑说明**：如果 $B$ 与 $A$ 语义极度相似（$\cos$ 高），且 $B$ 的长度足以容纳 $A$ 的大部分信息（长度比高），则 $ASF$ 趋近于 1，底层细节 $A$ 可被深度休眠；反之，若 $B$ 只是极短的浓缩摘要，$A$ 的核心物理细节仍被系统保留。
- **工程存储**：每个基本事件维护一个累积遮蔽因子 `ASF_i`（初始为 0），存入 Tier-1 元数据库，供 §4.5 复合权重计算使用。

#### 3.4.2 多层级嵌套抽象的链式传导算法（DAG Propagation）

当产生"抽象事件的再次抽象（如抽象 $C$ 包含 抽象 $B$，抽象 $B$ 包含 基础 $A$）"时，执行三步法传导：

**第一步：独立生存率连乘**
概率更新不能做减法（否则多次遮蔽后可能变为负值）。节点原有生存率为 $S_{old}$，被上层传入的遮蔽力 $ASF_{incoming}$ 抑制后：
$$S_{new} = S_{old} \times (1 - ASF_{incoming})$$

**第二步：LLM 动态穿透衰减常数（$\lambda$）**
高阶抽象向下穿透的遮蔽力逐层递减。在生成高阶抽象时，由 LLM 动态评估输出一个"对底层事实的覆写强度系数 $\lambda \in [0, 1]$"：

- $\lambda = 1.0$：高阶抽象完全覆盖底层细节（强遮蔽）；
- $\lambda = 0.0$：高阶抽象只是概念标记，底层事实独立存活（无遮蔽）。

实际下发的遮蔽力：
$$ASF_{propagated} = ASF_{incoming} \times \lambda$$

**第三步：递归更新**
沿着 DAG（有向无环图）拓扑，将 $ASF_{propagated}$ 递归下发并更新所有底层的基本叶子事件。工程上可通过 BFS 遍历 source_events 链路完成。

**最终效果**：频繁被抽象囊括的底层事件，其在 Tier-1 活跃池中的存活权重逐步降低，减少与高阶摘要的重复召回，实现信息总量的守恒。

