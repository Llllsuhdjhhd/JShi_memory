# 匠石对接说明：长期经验模块

- 状态：对接草稿；以本页“当前可用”部分为现行程序边界
- 日期：2026-09-27
- 实现：`src/rems/experience/`
- 相关设计：[经验模块程序框架](experience-module.md)、[经验形成与经验召回框架](experience.md)

## 一句话说明

长期经验模块让匠石在遇到某个人或某种情境时，获得有来源、可修订的相处经验或一般经验。它和回忆具体 Event 并行：Event 回答“发生过什么”，经验提供“目前怎样理解与应对”。经验是可被新证据修正的认识，不能当作对人物或世界的绝对事实。

## 名词与归属

| 内容 | 存放与用途 |
| --- | --- |
| Event 白描 | REMS 已有的单次事件人物材料；可用于具体回忆，也是人物经验的证据。 |
| 人物肖像 | REMS 的描述性人物资料；白描整理形成，按 L1 到最多 L10 提供不同细节量。现有匠石 `portrait()` 对应此项。 |
| 人物经验肖像 | REMS 新增的简短相处经验；按人物 `object_id` 定位，内容是关系基调、条件性相处调整和不确定处。它没有 L 级。 |
| 通用经验 | 不限于人物的长期认识；由未来匠石侧 Dream 等认知过程提出候选，交给 REMS 经验模块校验和保存。 |

匠石现有 `MemoryExperience` / `ExperienceLedger` 表示活动片段及其写入流水，与本文的长期经验记录不是同一数据类型。对接时应另设清楚的接口或类型名，避免混用。

## 当前可用的 REMS 入口

调用方需要使用同一个 REMS 数据库构造 `RoleRepository` 与 `PersonExperienceRepository`；数据库表由 `Database.create_tables()` 建立。`ExperienceModule.from_repositories(config, white_paintings, portraits, llm=...)` 是当前装配入口。传入 `llm` 才能生成或更新人物经验；未传入时仍可读取已有肖像，但待处理白描不会被模型提炼。该模块**尚未**挂入 `REMSPipeline`、HTTP API 或匠石现有 `MemoryBackendPort`。

### 1. 人物经验维护

```python
result = experience_module.refresh_person(subject_id, object_id)
```

调用会导入该人物已存储但尚未进入经验模块的白描，按来源 Event 去重。待处理白描达到 `experience_person_trigger_count`（默认 10 条）**或** `experience_person_trigger_chars`（默认 1200 字）时，模型才尝试更新。一次调用只处理一个 `object_id`。当前没有按时间自动触发，也没有批量扫描或后台定时任务；调用方需要负责何时调用及如何枚举人物。

`ExperiencePortraitSyncResult` 返回 `object_id`、`imported_white_paintings`、`updated`、`pending_white_paintings`、`pending_chars` 和 `review_outcome`。`review_outcome` 可能是 `waiting`（未达阈值或没有模型）、`already_reviewed`（同一批证据曾判断不足）、`insufficient`、`invalid`、`initial`、`confirm`、`extend`、`revise`、`uncertain`。`confirm` 表示评估过新白描但肖像内容未改变；它会消耗本次已评估的待处理白描。`insufficient` 不生成肖像，后续有新增白描时才重新评估。

人物经验肖像默认完整文本最多 400 字，短版本最多 120 字，可通过 REMS 配置修改。修订历史保留判断类型、原因、本轮来源 Event ID 和快照；`revise` / `uncertain` 的每条修改记录含 `previous`、`updated`、`reason`。当前证据追溯到**一次肖像修订**，尚未精确关联到每条条件性调整。

### 2. 经验召回

```python
result = experience_module.recall(ExperienceRecallQuery(
    subject_id=subject_id,
    query=current_text,
    object_ids=(speaker_object_id,),
    budget_chars=500,
    person_limit=1,
    general_limit=0,
))
```

`ExperienceRecallResult` 分别返回 `person` 和 `general`，并提供 `used_chars`。当前人物路需要调用方提供**已识别的相关 `object_id`**；模块不会从文本中自动识别人名。人物结果 `PersonExperienceFragment` 含 `type="person_experience"`、`object_id`、`content`、`variant`（`full` 或 `brief`）、`source_event_ids`、`updated_at`、`score`。完整文本放得下就选完整文本，否则尝试短版本；两者都放不下则不返回。找不到该人物、主体不匹配或尚未形成经验肖像时，返回空列表。

当前通用经验路只有接口与结果形状。若未提供 `general_search`，`general` 为空；**通用经验持久化库和 Dream 生成流程尚未实现**。召回只会返回 `established` 状态的通用经验，不返回 `tentative`。人物与通用经验共享 `budget_chars`，预算是内容字符数，不含对话上下文的其他开销。

下面是便于匠石适配器理解的**示意序列化**；当前 Python 入口实际返回 dataclass，并未定义 HTTP JSON 协议：

```json
{
  "person": [
    {
      "type": "person_experience",
      "object_id": "OBJ-xiaomi",
      "content": "与小米谈事务时可以直接说清要点；在她专注工作时，先简短说明来意，减少连续追问。",
      "variant": "full",
      "source_event_ids": ["EVT-1", "EVT-2"],
      "score": 1.0
    }
  ],
  "general": []
}
```

这里的 `source_event_ids` 表示该人物经验肖像所吸收的全部白描来源 Event，并不宣称每个 Event 都支持文本里的每一句话。

### 3. 留给 Dream 的通用经验候选

`ExperienceCandidate` 含 `core`、`evidence_event_ids`、可选 `applicable_context`、`action_or_expectation`、`observed_outcome`、`exceptions`，以及 `status`（`tentative` 或 `established`）。提交时必须提供有效原始 Event 来源；`established` 还必须提供适用情境和已观察结果。当前校验入口不会从事件中自动提取候选，也不会替 Dream 决定某条经验是否值得形成。通用经验存储由调用方注入，REMS 尚无可直接使用的持久化实现。

## 建议匠石侧怎样接

1. 在 REMS 适配器旁增加**独立的长期经验读取端口**，把 `ExperienceRecallQuery` 和结果映射为匠石侧稳定的数据类型。保留现有 `recall()` 的 Event 语义、现有 `portrait()` 的描述性人物肖像语义，也保留 `ingest_batch` / `record` 流程。
2. 在匠石已识别当前对话对象之后，使用该对象的稳定 `object_id` 查询人物经验。把返回内容作为“可修订的相处经验”交给上下文组装器；携带 `type`、`object_id`、来源 Event ID，避免将其显示成具体往事或客观人物事实。上下文需要限制篇幅，未命中时直接跳过。
3. 将人物经验维护安排在非对话关键路径：由匠石的调度/空闲机制或独立任务调用 REMS 的 `refresh_person`。首次上线可以只维护已识别且有白描的人物；需要先决定调用方如何枚举这些人物。目前周期调用本身不会绕过条数/字数阈值。
4. Dream 后续放在匠石侧，负责回顾、召回、比较与提出不同类型的认知结果。其通用经验候选可通过 `submit_dream_candidates(subject_id, candidates)` 交给 REMS；只有同时配置 `events` 与 `general_store` 才能使用该入口。它校验来源 Event，不承担 Dream 的思考过程。其他 Dream 结果应由其各自的匠石模块处理。

建议新增的匠石端口可以表达为 `recall_experience(subject_id, query, object_ids, budget_chars)` 与 `refresh_person_experience(subject_id, object_id)`；这是**对接建议**，不是已经存在的匠石 API。第一阶段可先显式刷新少量人物并接入读取，随后安排后台维护。不要把经验文本无条件塞入每次输入：只有相关人物或情境命中时才装配。

## 对接验收场景

- 某人物只有少量白描时，维护调用返回 `waiting`，对话召回为空；仍可按原路径回忆 Event。
- 白描达到阈值并形成肖像后，提供该人物 `object_id` 可召回 `person_experience`；小预算会选 `brief`，无关人物不会命中。
- 新白描只确认旧认识时，返回 `confirm`，版本与内容不变；发生修订时版本递增，修订记录保留本轮来源 Event。
- 没有配置通用经验库时，人物经验仍可独立召回，`general` 为空。
- 匠石现有 Event 召回、描述性人物肖像和记忆写入路径继续按各自语义运行。

## 尚需另行设计

匠石侧稳定端口及上下文装配位置、人物定期维护调度、人物识别与多人物预算分配、通用经验持久化检索、Dream 的运行机制、按单条相处经验追溯证据，以及真实模型案例下的经验质量评估，都不属于当前已接通的功能。
