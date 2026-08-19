"""Minimal pipeline ingest: recall_log rows match RecallBlock order and de-duplication."""

from __future__ import annotations

from datetime import datetime, timedelta

from rems.models.event import Event, EventRoleEntry, EmotionalModel, Importance, RoleSnapshot
from rems.models.role import Role
from rems.models.role import WhitePaintingEntry
from rems.pipeline import REMSPipeline, ProcessingMode


def _seed_recallable_event(event_repo, role_repo, vector_store, *, role_id: str) -> str:
    now = datetime.now()
    entry = EventRoleEntry(
        role_id=role_id,
        importance=Importance.B,
        role_snapshot=RoleSnapshot(l2_interaction="会议发言"),
        emotional_model=EmotionalModel(),
    )
    ev = Event(
        content_raw="张三在北京项目组讨论接口契约",
        summaries={
            "L1": "张三讨论接口契约 L1",
            "L2": "张三讨论契约 L2",
            "L3": "张三会议 L3",
        },
        role_list=[entry],
        create_time=now - timedelta(minutes=5),
    )
    event_repo.save(ev)
    vector_store.add_event(
        ev.event_id,
        "张三讨论接口契约",
        {
            "is_abstract": False,
            "create_time": ev.create_time.timestamp(),
            f"role_{role_id}": True,
            "role_ids": [role_id],
        },
    )
    wp = WhitePaintingEntry(
        event_id=ev.event_id,
        role_summary="张三会议上发言",
        forgetting_factor=1.0,
        memory_weight=0.5,
        create_time=ev.create_time,
        last_accessed_time=now,
    )
    role_repo.add_white_painting_entry(role_id, wp)
    vector_store.add_white_painting(role_id, ev.event_id, wp.role_summary, {"create_time": now.timestamp()})
    return ev.event_id


def test_ingest_recall_log_matches_recall_block(config, db, tmp_dir):
    pipeline = REMSPipeline.from_config(config)
    pipeline.role_skill = None
    pipeline.metabolism_service.process_input = lambda *a, **kw: []  # type: ignore[method-assign]
    pipeline.abstraction_service.mine_and_synthesize = lambda: []  # type: ignore[method-assign]

    rid = "ROL-pipe-zhang"
    pipeline.role_repo.save(Role(role_id=rid, name="张三"))
    eid = _seed_recallable_event(
        pipeline.event_repo,
        pipeline.role_repo,
        pipeline.vector_store,
        role_id=rid,
    )

    result = pipeline.ingest("张三那次会议说了什么 about 接口", mode=ProcessingMode.DIALOGUE)
    assert result.context_package is not None
    block = result.context_package.recall_block
    assert block.items, "expected recall to surface seeded basic event"

    ordered = [it.event_id for it in block.items]
    ordered_dedup: list[str] = []
    seen: set[str] = set()
    for e in ordered:
        if e in seen:
            continue
        seen.add(e)
        ordered_dedup.append(e)

    rows = pipeline.recall_log_repo.list_all()
    assert rows, "recall_log should record this ingest"
    _, logged = rows[-1]
    assert logged == ordered_dedup
    assert eid in logged
