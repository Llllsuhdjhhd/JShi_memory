# 经验模块：人物相处经验、Dream 边界与双路召回

- 状态：框架验证中
- 更新日期：2026-09-27
- 设计原则：[经验形成与经验召回框架](experience.md)
- 程序位于 `src/rems/experience/`；草稿目录只存文档。

## 三层概念

| 层次 | 内容 | 用途 |
| --- | --- | --- |
| 事件白描 | 单次事件中关于某个人的原始事实材料，绑定 Event | 回忆具体发生过什么；也是后两层的来源 |
| 人物肖像 | 对多条白描的整理、润色和归纳，保留较丰富的人物信息 | 需要了解或回忆这个人时，按 L 级选择细节量；完整级默认最多 8000 字 |
| 人物经验肖像 | 长期相处后形成的简短关系基调与相处方式 | 当前遇到此人时，快速进入合适的互动状态 |

三者都应保留，人物肖像与经验肖像分别存储、分别召回。经验肖像不是人物肖像的最高压缩级，也不承担具体往事的复述。例如两条白描是「小米专注工作时，对模糊问题回应很短」「简短明确的事务提醒没有打断她」，经验肖像可以是「与小米谈事务时，先说清要点；她专注时减少连续追问」。这仍只是有证据支持的情境性倾向，不是永久性格标签。证据不足时暂不形成经验肖像。

## 已搭建的程序边界

1. `WhitePaintingExperiencePortraitBuilder.sync` 显式读取已有白描，按来源 Event 去重，写入经验模块专用的待处理证据表。不会触碰现有 `ingest_batch`、`record`、人物肖像摘要流程。
2. 每人待处理白描达到 `experience_person_trigger_count` 条或 `experience_person_trigger_chars` 字时，调用 `PersonExperienceComposerPort`。默认阈值分别为 10 条与 1200 字；可配置。没有提炼器时保留证据；明确判断证据不足时记录已评估的证据批次，新增白描前不重复调用模型。
3. `LLMPersonExperienceComposer` 把旧结构化经验肖像与新增白描交给模型。所有非证据不足的判断都须返回完整快照，包括 `confirm`；已有肖像而新证据不足以改变认识时使用 `confirm`，不能返回证据不足。输出关系基调、附适用情境的相处调整、不确定处和短版本；完整文本由结构字段渲染。提示词要求保留未被新证据挑战的旧认识，并区分相处经验、人物事实与人格标签。空值、长度、整条白描原样复制和修订声明会校验；语义保留和证据判断仍需真实模型评估。
4. 经验肖像存于独立的 `person_experience_portraits` 和 `person_experience_evidence` 表；`person_experience_revisions` 追加每次形成、确认或修订的证据与快照，`person_experience_reviews` 记录证据不足的批次。`object_portraits` / `portrait_summaries` 继续服务人物肖像，两者互不覆盖。
5. 经验肖像默认最多 400 字，另有最多 120 字的短版本供小预算召回；这不是 L 级，也没有分块。更新分为 `initial`、`confirm`、`extend`、`revise`、`uncertain`。`confirm` 必须完整返回与旧肖像相同的结构；确认不变时消耗已评估白描并保留旧版本。`revise` 或 `uncertain` 须说明原因，每项 `revised_claims` 均记录 `previous`、`updated`、`reason`。所有成功评估都留下追加式修订记录。
6. `ExperiencePortraitReader` 仅召回本模块形成的经验肖像，按 `subject_id`、`object_id` 定位；根据预算选完整或短版本，并附上来源 Event ID。当前相关人物由调用方提供；自动识别与相关性评分尚未接入。
7. `ExperienceModule.recall` 分别调用人物经验与通用经验检索，返回两组有类型的结果，控制总字符数。通用经验检索依赖将来的持久化索引适配器；未配置时为空。
8. `DreamCandidateSink` 仅为未来 Dream 提供结构化候选提交接口，并核验来源 Event。候选区分 `tentative`（待验证）与 `established`（正式经验）；正式经验必须有适用情境和观察结果。Dream 的回顾、联想、归纳和修订算法没有在本轮实现。

这些经验模块入口目前须由调用方显式装配和调用，尚未接入主编排或 HTTP API。相邻匠石项目的 REMS 适配器虽然提供 `portrait()` 转发，但检索和上下文装配代码只调用 `recall()`；没有找到匠石侧调用人物肖像或经验肖像的代码。REMS 主 `recall()` 当前也只走事件检索，经验双路召回尚未并入。

## 经验记录的形状

人物经验肖像的内部字段是 `baseline`、若干 `adjustments(context, approach)`、若干 `uncertainties`；对外使用完整文本和短版本。模型每次输出新快照与修订类型，仓储保存旧版修订记录及本次白描来源 Event。这个结构表达「怎么自然相处」，不承担人物传记。提示词要求未被新证据挑战的旧认识默认保留；程序能严格校验 `confirm` 未改动和 `extend` 未删去旧调整或不确定项，但其他类型的语义漂移仍需实例评估，不能仅靠字段验证。

通用经验候选含 `core`（核心认识）、`applicable_context`（适用情境）、可选的 `action_or_expectation`（行动或预期）、`observed_outcome`（观察结果）、`exceptions`（例外）、来源 Event ID 和状态。`tentative` 可以保留尚缺完整条件的线索，但不进入正式召回；`established` 须有情境、结果及有效来源。该形状已落实到候选与结果类型，持久化经验库仍待下一阶段。

## 人物肖像的独立刷新

新增 `WhitePaintingPortraitBuilder.refresh_due(subject_id)` 作为周期扫描入口，它枚举该主体有白描的人物并逐个检查；`refresh(subject_id, object_id)` 可单独刷新一个人。它不挂在每次事件封存或每条白描写入上。调用时读取已存储的白描，只把新增白描放进人物肖像待处理区。同一来源 Event 不会重复并入。触发条件可配置，默认待处理白描达到 10 条，或有待处理白描且距上次肖像更新（首次则距最早一条白描）已达 24 小时。两者满足一个即可；没有新白描不更新。`force=True` 供显式人工刷新。当前只提供批量入口，外部定时调度尚未接入。

有模型时，使用旧人物肖像最完整版本和新增白描形成新的 L1；L1 最多 8000 字。L2 基于 L1 压缩，L3 基于 L2 继续压缩，以此类推，最多 L10。每一级若达到与事件摘要相同的 `summary_fuse_min_chars` 阈值，就在该级停止；模型不能有效缩短时也停止。实际结果可以只有 L1，也可以在 L5 等级结束。无模型或 L1 无效时保留待处理白描，不用拼接原文冒充整理后的肖像。人物肖像仍存于 `object_portraits` / `portrait_summaries`，经验肖像仍用自己的表。

```python
from rems.services.white_painting_portrait import WhitePaintingPortraitBuilder
from rems.storage.repository import PortraitRepository, RoleRepository

builder = WhitePaintingPortraitBuilder(
    config, RoleRepository(db), PortraitRepository(db), llm,
)
# 由周期任务调用；一次扫描该主体所有有白描的人物。
builder.refresh_due("jshi-1")
```

已删除未接入主流程的反向 L 生成路径；人物肖像只有白描批量刷新这一条生成路径。`PortraitService` 保留仓储读取入口，现有 `portrait()` 可读取新肖像。匠石侧是否实际调用该接口、以及如何把人物肖像与经验肖像装配到上下文，留待后续接入。

```python
from rems.experience import ExperienceModule, ExperienceRecallQuery, PersonExperienceRepository
from rems.storage.repository import RoleRepository

module = ExperienceModule.from_repositories(
    config,
    RoleRepository(db),
    PersonExperienceRepository(db),
    llm=llm,
)
module.refresh_person("jshi-1", "OBJ-xiaomi")
result = module.recall(ExperienceRecallQuery(
    subject_id="jshi-1", query="现在怎样提醒小米？", object_ids=("OBJ-xiaomi",),
))
```

## Dream 的后续方向

Dream 可能定期回顾当天或最近几天 Event，发现值得追问的线索，再回忆更早的相关经历，比较情境、例外和结果，形成零条或多条经验候选。候选须带来源 Event ID；经验库独立于抽象 Event。近期窗口、历史召回、冲突修订、成本控制都留给 Dream 的专门设计，不能用任意几个 Event 的简单提取代替。

`ExperienceCandidate` 按上一节的结构通过 `ExperienceModule.submit_dream_candidates` 提交。经验模块负责字段与证据核验，再交给注入的通用经验存储；通用经验持久化索引尚未实现。

## 召回方式

人物路通过当前相关的 `object_id` 读取经验肖像；通用路通过当前问题搜索经验库，且只返回 `established`。两路各自筛选候选，再按总预算装配，返回 `person_experience` 与 `general_experience`。具体事件回忆仍是独立入口。Jev 若将来用于经验召回，也只针对粗召回后的有限候选；当前未接入。

当前人物经验路是对象定位加可选相关性评分，尚不能从输入自动判断哪个人相关。完整/短版本只解决经验肖像的预算选择。真实使用中应验证它是否能在新情境下帮助互动，而不是只生成貌似合理的忠告。

## 验证与待讨论

`tests/unit/test_experience_framework.py` 覆盖触发阈值、证据不足去重评估、初次形成、修订、确认、不确定、主体与人物肖像隔离、Dream 证据校验、正式/待验证状态和双路预算。它验证程序契约，不证明模型提炼出的相处经验足够准确。

待讨论：如何接入实际周期调度与匠石侧上下文；如何用真实案例评估经验肖像的相处效果与修订质量；何时让旧认识失效；多人物同时出现时如何分配预算；何时由 Dream 对人物经验提出补充或纠正。
