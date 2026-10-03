"""The new experience entry points stay separate from Event ingest and recall."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from rems.experience import (
    ExperienceCandidate, ExperienceModule, ExperienceRecallQuery,
    PersonExperienceRepository,
)
from rems.models.event import Event
from rems.models.portrait import ObjectPortrait, PortraitLevelInfo
from rems.models.role import WhitePaintingEntry
from rems.storage.repository import EventRepository, PortraitRepository, RoleRepository
from rems.storage.database import Database
from rems.experience.models import ContextualAdjustment
from rems.experience.person_composer import render_person_experience


def test_render_person_experience_strips_uncertainty_terminal_punctuation():
    assert render_person_experience(
        "先核对证据。",
        (),
        ("是否适用于别的场景尚无证据。",),
    ) == "先核对证据；尚不确定：是否适用于别的场景尚无证据。"


def test_render_person_experience_does_not_duplicate_context_suffix():
    assert render_person_experience(
        "按情境回应。",
        (
            ContextualAdjustment("在值班操作时", "给三条短步骤。"),
            ContextualAdjustment("倾诉心情", "先听，不列清单。"),
        ),
        (),
    ) == "按情境回应；在值班操作时，给三条短步骤；在倾诉心情时，先听，不列清单。"


class MemoryGeneralStore:
    """A bounded-search stand-in; persistent indexing belongs to later work."""

    def __init__(self):
        self.items = []

    def save(self, item):
        self.items.append(item)

    def search(self, subject_id, query, *, limit):
        return tuple(
            (item, 1.0)
            for item in self.items
            if item.subject_id == subject_id and query in item.core
        )[:limit]


def test_white_paintings_form_interaction_experience_after_threshold(config, db, fake_llm):
    config.experience_person_trigger_count = 2
    config.experience_person_trigger_chars = 10000
    white_paintings = RoleRepository(db)
    portraits = PersonExperienceRepository(db)
    legacy = PortraitRepository(db)
    legacy.save_portrait(ObjectPortrait(
        subject_id="jshi-1",
        object_id="OBJ-X",
        levels={"L1": PortraitLevelInfo(
            level="L1", text="旧描述性肖像", budget=100, actual_len=7,
        )},
        max_level="L1",
    ))
    white_paintings.add_white_painting_entry(
        "OBJ-X",
        WhitePaintingEntry(
            subject_id="jshi-1",
            event_id="EVT-1",
            role_summary="小米专注工作时，对模糊问题回应很短。",
        ),
    )
    module = ExperienceModule.from_repositories(
        config, white_paintings, portraits, llm=fake_llm,
    )

    first = module.refresh_person("jshi-1", "OBJ-X")
    assert first.imported_white_paintings == 1 and not first.updated
    assert first.pending_white_paintings == 1
    assert portraits.get_portrait("OBJ-X") is None
    assert legacy.get_portrait("OBJ-X").levels["L1"].text == "旧描述性肖像"

    white_paintings.add_white_painting_entry(
        "OBJ-X",
        WhitePaintingEntry(
            subject_id="jshi-1",
            event_id="EVT-2",
            role_summary="简短明确的事务提醒没有打断小米的工作。",
        ),
    )
    fake_llm.push_response({"insufficient_evidence": True})
    pending = module.refresh_person("jshi-1", "OBJ-X")
    assert not pending.updated and pending.pending_white_paintings == 2
    assert pending.review_outcome == "insufficient"
    assert portraits.get_portrait("OBJ-X") is None

    calls_after_review = len(fake_llm.calls)
    assert module.refresh_person("jshi-1", "OBJ-X").review_outcome == "already_reviewed"
    assert len(fake_llm.calls) == calls_after_review

    white_paintings.add_white_painting_entry(
        "OBJ-X",
        WhitePaintingEntry(
            subject_id="jshi-1", event_id="EVT-3",
            role_summary="小米愿意在工作间隙讨论较长的话题。",
        ),
    )

    fake_llm.push_response({
        "insufficient_evidence": False,
        "revision_kind": "initial",
        "revised_claims": [],
        "baseline": "和小米交流可以直接、明确。",
        "adjustments": [{
            "context": "她高度专注工作",
            "approach": "先给出简短明确的事务信息，避免连续追问",
        }],
        "uncertainties": ["较长话题是否总适合在工作间隙谈"],
        "brief": "和小米沟通先简短明确，专注时少打断。",
    })
    second = module.refresh_person("jshi-1", "OBJ-X")
    third = module.refresh_person("jshi-1", "OBJ-X")
    recalled = module.recall(ExperienceRecallQuery(
        subject_id="jshi-1", query="小米", object_ids=("OBJ-X",), budget_chars=200,
    ))

    assert second.imported_white_paintings == 1 and second.updated
    assert second.review_outcome == "initial"
    assert portraits.get_portrait("OBJ-X").baseline == "和小米交流可以直接、明确。"
    assert len(portraits.get_portrait("OBJ-X").adjustments) == 1
    assert third.imported_white_paintings == 0 and not third.updated
    assert len(portraits.list_summaries("OBJ-X")) == 3
    assert tuple(item.kind for item in portraits.list_revisions("jshi-1", "OBJ-X")) == ("initial",)
    assert len(recalled.person) == 1
    assert recalled.person[0].source_event_ids == ("EVT-1", "EVT-2", "EVT-3")
    assert recalled.person[0].type == "person_experience"
    assert "简短明确" in recalled.person[0].content
    assert "模糊问题回应很短" not in recalled.person[0].content
    assert legacy.get_portrait("OBJ-X").levels["L1"].text == "旧描述性肖像"
    assert module.recall(ExperienceRecallQuery(
        subject_id="other", query="小米", object_ids=("OBJ-X",),
    )).person == ()
    with pytest.raises(ValueError, match="different subject"):
        module.refresh_person("other", "OBJ-X")


def test_descriptive_portrait_budget_is_independent(config):
    assert config.portrait_max_chars == 8000
    assert config.portrait_max_levels == 10
    assert config.experience_person_max_chars < config.portrait_max_chars


def test_existing_experience_portrait_table_gains_structured_columns(tmp_path):
    url = f"sqlite:///{tmp_path / 'old-experience.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE person_experience_portraits ("
            "object_id VARCHAR PRIMARY KEY, subject_id VARCHAR NOT NULL, "
            "content TEXT NOT NULL, brief TEXT NOT NULL, "
            "total_content_len INTEGER, created_at DATETIME NOT NULL, "
            "updated_at DATETIME NOT NULL)"
        ))
        connection.execute(text(
            "INSERT INTO person_experience_portraits "
            "(object_id, subject_id, content, brief, total_content_len, created_at, updated_at) "
            "VALUES ('OBJ-OLD', 'jshi-1', '旧经验肖像', '旧经验', 20, "
            "'2026-09-27 12:00:00', '2026-09-27 12:00:00')"
        ))
    engine.dispose()

    database = Database(url)
    database.create_tables()
    columns = {
        column["name"] for column in inspect(database.engine).get_columns(
            "person_experience_portraits"
        )
    }
    assert {"shape", "version"} <= columns
    assert PersonExperienceRepository(database).get_portrait("OBJ-OLD").baseline == "旧经验肖像"


def test_dream_boundary_validates_evidence_and_general_recall(config, db):
    events = EventRepository(db)
    good = Event(subject_id="jshi-1", content_raw="雨后路滑")
    foreign = Event(subject_id="other", content_raw="别人的经历")
    events.save(good)
    events.save(foreign)
    store = MemoryGeneralStore()
    module = ExperienceModule.from_repositories(
        config,
        RoleRepository(db),
        PersonExperienceRepository(db),
        events=events,
        general_store=store,
        general_search=store,
    )

    with pytest.raises(ValueError, match="invalid evidence"):
        module.submit_dream_candidates("jshi-1", (
            ExperienceCandidate(
                "雨后路滑", (good.event_id,),
                applicable_context="雨后山洞", observed_outcome="走路更容易打滑",
                status="established",
            ),
            ExperienceCandidate("错误证据", (foreign.event_id,)),
        ))
    assert store.items == []

    ids = module.submit_dream_candidates("jshi-1", (
        ExperienceCandidate(
            "雨后山洞左侧路滑", (good.event_id,),
            applicable_context="雨后进山洞",
            action_or_expectation="先确认左侧地面",
            observed_outcome="左侧曾观察到湿滑",
            exceptions=("干燥天气尚未验证",),
            status="established",
        ),
    ))
    recalled = module.recall(ExperienceRecallQuery(
        subject_id="jshi-1", query="雨后", budget_chars=100,
    ))

    assert len(ids) == 1 and ids[0].startswith("EXP-")
    assert len(recalled.general) == 1
    assert recalled.general[0].evidence_event_ids == (good.event_id,)
    assert recalled.general[0].observed_outcome == "左侧曾观察到湿滑"
    assert recalled.general[0].exceptions == ("干燥天气尚未验证",)
    assert recalled.used_chars <= 100
    assert module.recall(ExperienceRecallQuery(
        subject_id="other", query="雨后", budget_chars=100,
    )).general == ()

    module.submit_dream_candidates("jshi-1", (
        ExperienceCandidate("雨后也许要绕路", (good.event_id,)),
    ))
    assert module.recall(ExperienceRecallQuery(
        subject_id="jshi-1", query="也许", budget_chars=100,
    )).general == ()

    with pytest.raises(ValueError, match="context and observed outcome"):
        module.submit_dream_candidates("jshi-1", (
            ExperienceCandidate("缺少结果", (good.event_id,), status="established"),
        ))


def test_person_experience_does_not_fall_back_to_white_painting(config, db):
    config.experience_person_trigger_count = 1
    config.experience_person_trigger_chars = 0
    paintings = RoleRepository(db)
    portraits = PersonExperienceRepository(db)
    paintings.add_white_painting_entry(
        "OBJ-Y",
        WhitePaintingEntry(
            subject_id="jshi-1", event_id="EVT-Y",
            role_summary="这个人当时只说了一个字。",
        ),
    )
    module = ExperienceModule.from_repositories(config, paintings, portraits)

    result = module.refresh_person("jshi-1", "OBJ-Y")

    assert result.pending_white_paintings == 1
    assert not result.updated
    assert portraits.get_portrait("OBJ-Y") is None


def test_person_experience_revision_keeps_history_and_evidence(config, db, fake_llm):
    config.experience_person_trigger_count = 1
    config.experience_person_trigger_chars = 0
    paintings = RoleRepository(db)
    portraits = PersonExperienceRepository(db)
    module = ExperienceModule.from_repositories(
        config, paintings, portraits, llm=fake_llm,
    )

    def add(event_id, text):
        paintings.add_white_painting_entry("OBJ-A", WhitePaintingEntry(
            subject_id="jshi-1", event_id=event_id, role_summary=text,
        ))

    add("E1", "阿青与我从小一起上学，见面后能直接聊天。")
    fake_llm.push_response({
        "revision_kind": "initial",
        "revised_claims": [],
        "baseline": "阿青是熟悉的老同学，相处可以直接。",
        "adjustments": [], "uncertainties": [],
        "brief": "与阿青是熟悉的老同学。",
    })
    assert module.refresh_person("jshi-1", "OBJ-A").review_outcome == "initial"
    assert portraits.get_portrait("OBJ-A").version == 1

    add("E2", "阿青不愿在众人面前谈私事。")
    fake_llm.push_response({
        "revision_kind": "revise",
        "revision_reason": "新增白描显示私事需要单独交谈",
        "revised_claims": [{
            "previous": "任何话题都可直接谈",
            "updated": "私事留到单独交谈",
            "reason": "公开场合谈私事出现反例",
        }],
        "baseline": "阿青是熟悉的老同学，相处可以直接，私事除外。",
        "adjustments": [{"context": "涉及私事且有其他人在场", "approach": "留到单独交谈"}],
        "uncertainties": [],
        "brief": "与阿青相处直接，私事宜单独谈。",
    })
    revised = module.refresh_person("jshi-1", "OBJ-A")
    portrait = portraits.get_portrait("OBJ-A")
    assert revised.updated and revised.review_outcome == "revise"
    assert portrait.version == 2
    assert portrait.adjustments[0].approach == "留到单独交谈"
    revisions = portraits.list_revisions("jshi-1", "OBJ-A")
    assert revisions[-1].evidence_event_ids == ("E2",)
    assert revisions[-1].revised_claims[0].updated == "私事留到单独交谈"
    assert revisions[0].baseline == "阿青是熟悉的老同学，相处可以直接。"
    assert revisions[0].adjustments == ()

    # 已有肖像时「证据不足」= 本批不改，白描算已吸收。
    add("E3", "阿青再次建议有些私事另找时间单独谈。")
    fake_llm.push_response({"insufficient_evidence": True})
    confirmed = module.refresh_person("jshi-1", "OBJ-A")
    assert not confirmed.updated and confirmed.review_outcome == "confirm"
    assert portraits.get_portrait("OBJ-A").version == 2
    assert portraits.list_summaries("OBJ-A", incorporated=False) == []

    # 缺字段是格式错误：记签名，同一批不再重试。
    add("E3b", "阿青说单独谈时更自在。")
    fake_llm.push_response({
        "revision_kind": "confirm",
        "revision_reason": "新白描支持现有的私下交谈边界",
    })
    assert module.refresh_person("jshi-1", "OBJ-A").review_outcome == "invalid"
    calls_after_invalid = len(fake_llm.calls)
    assert module.refresh_person("jshi-1", "OBJ-A").review_outcome == "already_reviewed"
    assert len(fake_llm.calls) == calls_after_invalid
    assert len(portraits.list_summaries("OBJ-A", incorporated=False)) == 1

    # 标签与内容不符：按实际差异改标签，草稿照收。
    add("E3c", "阿青约好下次单独喝茶。")
    fake_llm.push_response({
        "revision_kind": "confirm",
        "revision_reason": "标签写错，内容其实改了基调",
        "revised_claims": [],
        "baseline": "与阿青相处要保持距离。",
        "adjustments": [{"context": item.context, "approach": item.approach}
                        for item in portrait.adjustments],
        "uncertainties": list(portrait.uncertainties),
        "brief": portrait.brief,
    })
    relabeled = module.refresh_person("jshi-1", "OBJ-A")
    assert relabeled.updated and relabeled.review_outcome == "revise"
    assert portraits.get_portrait("OBJ-A").version == 3
    assert portraits.list_summaries("OBJ-A", incorporated=False) == []

    add("E4", "阿青最近主动在其他人在场时提起近况。")
    fake_llm.push_response({
        "revision_kind": "uncertain",
        "revision_reason": "近期表现与原边界不完全一致",
        "revised_claims": [{
            "previous": "有其他人在场时总不谈私事",
            "updated": "公开场合先观察是否适合继续",
            "reason": "出现主动谈近况的反例，但尚需区分话题",
        }],
        "baseline": "阿青是熟悉的老同学，相处可以直接。",
        "adjustments": [{"context": "涉及私事且有其他人在场", "approach": "先观察是否适合继续"}],
        "uncertainties": ["近期是否愿意公开谈某些近况，仍需区分话题"],
        "brief": "与阿青相处直接，公开场合谈私事先看情境。",
    })
    uncertain = module.refresh_person("jshi-1", "OBJ-A")
    assert uncertain.updated and uncertain.review_outcome == "uncertain"
    assert portraits.get_portrait("OBJ-A").version == 4
    assert tuple(item.kind for item in portraits.list_revisions("jshi-1", "OBJ-A")) == (
        "initial", "revise", "confirm", "revise", "uncertain",
    )

    add("E5", "阿青又一次调整了谈话场合。")
    fake_llm.push_response({
        "revision_kind": "revise",
        "revision_reason": "格式错误",
        "baseline": "阿青是熟悉的老同学。",
        "adjustments": "不是数组", "uncertainties": [],
        "brief": "与阿青熟悉。",
    })
    invalid = module.refresh_person("jshi-1", "OBJ-A")
    assert invalid.review_outcome == "invalid"
    assert invalid.pending_white_paintings == 1
    assert portraits.get_portrait("OBJ-A").version == 4

    # revise 没写 revised_claims 也照收，只是审计信息少一些。
    add("E5b", "阿青改在人少时说近况。")
    fake_llm.push_response({
        "revision_kind": "revise",
        "revision_reason": "改在人少时说近况",
        "baseline": "阿青是熟悉的老同学。",
        "adjustments": [], "uncertainties": [],
        "brief": "与阿青熟悉。",
    })
    accepted = module.refresh_person("jshi-1", "OBJ-A")
    assert accepted.review_outcome == "revise"
    assert portraits.get_portrait("OBJ-A").version == 5


def test_extend_only_adds_to_existing_person_experience(config, db, fake_llm):
    config.experience_person_trigger_count = 1
    config.experience_person_trigger_chars = 0
    paintings = RoleRepository(db)
    portraits = PersonExperienceRepository(db)
    module = ExperienceModule.from_repositories(
        config, paintings, portraits, llm=fake_llm,
    )
    paintings.add_white_painting_entry("OBJ-B", WhitePaintingEntry(
        subject_id="jshi-1", event_id="B1", role_summary="彼此认识多年。",
    ))
    fake_llm.push_response({
        "revision_kind": "initial", "baseline": "与阿白相处比较熟悉。",
        "revised_claims": [],
        "adjustments": [], "uncertainties": [], "brief": "与阿白很熟悉。",
    })
    assert module.refresh_person("jshi-1", "OBJ-B").updated

    paintings.add_white_painting_entry("OBJ-B", WhitePaintingEntry(
        subject_id="jshi-1", event_id="B2", role_summary="阿白在忙时希望先听简短结论。",
    ))
    # 标为 extend 却改了基调：按实际差异记为 revise。
    fake_llm.push_response({
        "revision_kind": "extend", "revision_reason": "忙碌情境的新白描",
        "revised_claims": [],
        "baseline": "与阿白相处应该保持距离。",
        "adjustments": [{"context": "阿白忙碌", "approach": "先说结论"}],
        "uncertainties": [], "brief": "阿白忙时先说结论。",
    })
    assert module.refresh_person("jshi-1", "OBJ-B").review_outcome == "revise"
    assert portraits.get_portrait("OBJ-B").version == 2

    paintings.add_white_painting_entry("OBJ-B", WhitePaintingEntry(
        subject_id="jshi-1", event_id="B3", role_summary="阿白开会前只要一句话。",
    ))
    # 标为 revise 却只增不改：按实际差异记为 extend。
    fake_llm.push_response({
        "revision_kind": "revise", "revision_reason": "开会前的新白描",
        "revised_claims": [],
        "baseline": "与阿白相处应该保持距离。",
        "adjustments": [
            {"context": "阿白忙碌", "approach": "先说结论"},
            {"context": "阿白开会前", "approach": "只说一句"},
        ],
        "uncertainties": [], "brief": "阿白忙时先说结论，开会前只说一句。",
    })
    result = module.refresh_person("jshi-1", "OBJ-B")
    assert result.updated and result.review_outcome == "extend"
    assert portraits.get_portrait("OBJ-B").version == 3
    assert tuple(item.kind for item in portraits.list_revisions("jshi-1", "OBJ-B")) == (
        "initial", "revise", "extend",
    )


def test_person_and_general_lanes_share_one_budget(config, db, fake_llm):
    config.experience_person_trigger_count = 2
    config.experience_person_trigger_chars = 10000
    white_paintings = RoleRepository(db)
    portraits = PersonExperienceRepository(db)
    events = EventRepository(db)
    event = Event(subject_id="jshi-1", content_raw="雨后探查山洞")
    events.save(event)
    white_paintings.add_white_painting_entry(
        "OBJ-X",
        WhitePaintingEntry(
            subject_id="jshi-1",
            event_id=event.event_id,
            role_summary="小米提醒雨后山洞左侧更滑。",
        ),
    )
    white_paintings.add_white_painting_entry(
        "OBJ-X",
        WhitePaintingEntry(
            subject_id="jshi-1",
            event_id="EVT-SECOND",
            role_summary="小米在危险路段更愿意先确认脚下情况。",
        ),
    )
    store = MemoryGeneralStore()
    module = ExperienceModule.from_repositories(
        config, white_paintings, portraits,
        llm=fake_llm,
        events=events, general_store=store, general_search=store,
    )
    fake_llm.push_response({
        "insufficient_evidence": False,
        "revision_kind": "initial",
        "revised_claims": [],
        "baseline": "和小米同行时可以共同确认路线。",
        "adjustments": [{
            "context": "进入不熟悉的路段",
            "approach": "先一起确认地面状况，再决定怎么走",
        }],
        "uncertainties": [],
        "brief": "与小米同行时先确认路况。",
    })
    assert module.refresh_person("jshi-1", "OBJ-X").updated
    module.submit_dream_candidates("jshi-1", (
        ExperienceCandidate(
            "雨后山洞左侧通常更滑", (event.event_id,),
            applicable_context="雨后进山洞",
            observed_outcome="左侧地面曾湿滑",
            status="established",
        ),
    ))

    recalled = module.recall(ExperienceRecallQuery(
        subject_id="jshi-1", query="雨后", object_ids=("OBJ-X",), budget_chars=240,
    ))
    assert len(recalled.person) == 1
    assert len(recalled.general) == 1
    assert recalled.used_chars <= 240


def test_backlog_is_absorbed_in_batches_and_length_is_only_a_target(config, db, fake_llm):
    config.experience_person_trigger_count = 1
    config.experience_person_trigger_chars = 0
    config.experience_person_batch_chars = 30
    config.experience_person_max_chars = 20
    paintings = RoleRepository(db)
    portraits = PersonExperienceRepository(db)
    module = ExperienceModule.from_repositories(config, paintings, portraits, llm=fake_llm)
    for index in range(5):
        paintings.add_white_painting_entry("OBJ-L", WhitePaintingEntry(
            subject_id="jshi-1", event_id=f"L{index}",
            role_summary=f"第{index}次：老李在忙时只想听结论。",
        ))
    long_baseline = "与老李相处直接，" + "先给结论" * 20
    fake_llm.push_response({
        "revision_kind": "initial", "revised_claims": [],
        "baseline": long_baseline, "adjustments": [], "uncertainties": [],
        "brief": "老李忙时先说结论，" + "不绕弯" * 20,
    })
    for index in (1, 2):
        fake_llm.push_response({
            "revision_kind": "extend", "revision_reason": f"第{index}批补充忙碌情境",
            "revised_claims": [], "baseline": long_baseline,
            "adjustments": [{"context": f"忙碌第{index}批", "approach": "先说结论"}]
            + [{"context": f"忙碌第{i}批", "approach": "先说结论"} for i in range(1, index)],
            "uncertainties": [], "brief": f"老李忙时先说结论（{index}）",
        })

    result = module.refresh_person("jshi-1", "OBJ-L")

    assert result.updated and result.pending_white_paintings == 0
    assert len(fake_llm.calls) == 3
    for call in fake_llm.calls:
        evidence = call["messages"][1]["content"].split("新增白描：", 1)[1]
        assert evidence.count("老李在忙时") <= 2
    portrait = portraits.get_portrait("OBJ-L")
    assert len(portrait.content) > config.experience_person_max_chars
    assert portrait.version == 3
    assert portraits.list_summaries("OBJ-L", incorporated=False) == []
