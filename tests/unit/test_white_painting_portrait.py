"""Descriptive portraits refresh in batches and stop compressing when short."""

from datetime import datetime, timedelta

from rems.models.role import WhitePaintingEntry
from rems.services.white_painting_portrait import WhitePaintingPortraitBuilder
from rems.storage.repository import PortraitRepository, RoleRepository


def test_count_trigger_builds_l1_then_stops_at_sufficiently_short_level(
    config, db, fake_llm
):
    config.portrait_white_painting_trigger_count = 2
    config.portrait_white_painting_trigger_hours = 24
    now = datetime(2026, 9, 27, 12)
    paintings = RoleRepository(db)
    portraits = PortraitRepository(db)
    builder = WhitePaintingPortraitBuilder(
        config, paintings, portraits, fake_llm, clock=lambda: now
    )
    paintings.add_white_painting_entry("OBJ-X", WhitePaintingEntry(
        subject_id="jshi-1", event_id="E1", role_summary="小米与我同班读书。",
        create_time=now,
    ))

    scan = builder.refresh_due("jshi-1")
    assert len(scan) == 1
    first = scan[0]
    assert not first.updated and first.pending_white_paintings == 1
    assert fake_llm.calls == []

    paintings.add_white_painting_entry("OBJ-X", WhitePaintingEntry(
        subject_id="jshi-1", event_id="E2", role_summary="多年以后仍能聊起共同的老师。",
        create_time=now,
    ))
    for length in (800, 350, 150, 50, 20):
        fake_llm.push_response({"text": "人" * length})
    result = builder.refresh("jshi-1", "OBJ-X")
    portrait = portraits.get_portrait("OBJ-X")

    assert result.updated and result.trigger == "count"
    assert result.levels == ("L1", "L2", "L3", "L4", "L5")
    assert portrait.max_level == "L1"
    assert portrait.longest_level.text == "人" * 800
    assert len(fake_llm.calls) == 5
    assert builder.refresh("jshi-1", "OBJ-X").updated is False
    assert len(portraits.list_summaries("OBJ-X")) == 2


def test_time_trigger_refreshes_only_when_new_white_painting_exists(
    config, db, fake_llm
):
    config.portrait_white_painting_trigger_count = 10
    config.portrait_white_painting_trigger_hours = 24
    now = datetime(2026, 9, 27, 12)
    paintings = RoleRepository(db)
    portraits = PortraitRepository(db)
    paintings.add_white_painting_entry("OBJ-Y", WhitePaintingEntry(
        subject_id="jshi-1", event_id="E1", role_summary="阿青与我一起上过学。",
        create_time=now - timedelta(hours=25),
    ))
    fake_llm.push_response({"text": "阿青是我过去的同学。"})
    builder = WhitePaintingPortraitBuilder(
        config, paintings, portraits, fake_llm, clock=lambda: now
    )

    result = builder.refresh("jshi-1", "OBJ-Y")

    assert result.updated and result.trigger == "time"
    assert result.levels == ("L1",)
    assert builder.refresh("jshi-1", "OBJ-Y").updated is False
    assert len(fake_llm.calls) == 1

    next_day = now + timedelta(hours=25)
    paintings.add_white_painting_entry("OBJ-Y", WhitePaintingEntry(
        subject_id="jshi-1", event_id="E2", role_summary="多年后仍记得阿青。",
        create_time=next_day,
    ))
    fake_llm.push_response({"text": "阿青是我过去的同学，多年后仍有联系。"})
    later = WhitePaintingPortraitBuilder(
        config, paintings, portraits, fake_llm, clock=lambda: next_day
    )

    second = later.refresh("jshi-1", "OBJ-Y")

    assert second.updated and second.trigger == "time"
    assert "阿青是我过去的同学。" in fake_llm.calls[-1]["messages"][1]["content"]
    assert portraits.get_portrait("OBJ-Y").longest_level.text.endswith("多年后仍有联系。")
