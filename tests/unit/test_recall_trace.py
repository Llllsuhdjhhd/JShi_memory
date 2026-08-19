"""Fine-grained recall tracing: tri-band search, scoring, tier selection."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rems.config import REMSConfig
from rems.embedding.tri_band import TriBandEncoder
from rems.models.event import Event, EventRoleEntry, EmotionalModel, Importance, RoleSnapshot
from rems.models.metabolism import Shadow
from rems.models.role import Role, WhitePaintingEntry
from rems.services.recall_service import RecallService
from rems.storage.repository import EventRepository, RoleRepository


def install_recall_trace(svc: RecallService) -> tuple[dict, callable]:
    trace: dict = {"tri_band_searches": [], "scored": []}
    vector = svc._vector
    orig_search = vector.search_tri_band

    def wrapped_search(*args, **kwargs):
        out = orig_search(*args, **kwargs)
        trace["tri_band_searches"].append({
            "n_hits": len(out),
            "filter_ids": kwargs.get("filter_ids"),
            "weights": kwargs.get("weights"),
        })
        return out

    vector.search_tri_band = wrapped_search  # type: ignore[method-assign]

    def uninstall():
        vector.search_tri_band = orig_search  # type: ignore[method-assign]

    return trace, uninstall


def _seed_event_with_wp(
    *,
    event_repo: EventRepository,
    role_repo: RoleRepository,
    vector_store,
    content: str,
    role_id: str,
    wp_summary: str,
    create_time: datetime | None = None,
    forgetting_factor: float = 1.0,
) -> Event:
    create_time = create_time or datetime.now()
    entry = EventRoleEntry(
        role_id=role_id,
        importance=Importance.B,
        role_snapshot=RoleSnapshot(l1_mention="mention", l2_interaction=wp_summary),
        emotional_model=EmotionalModel(),
    )
    ev = Event(
        content_raw=content,
        summaries={"L1": content[:80] + " L1", "L2": content[:50] + " L2", "L3": content[:30] + " L3"},
        role_list=[entry],
        create_time=create_time,
    )
    event_repo.save(ev)
    vector_store.upsert_event_vectors(ev)
    now = datetime.now()
    wp = WhitePaintingEntry(
        event_id=ev.event_id,
        role_summary=wp_summary,
        forgetting_factor=forgetting_factor,
        memory_weight=0.5,
        create_time=create_time,
        last_accessed_time=now,
    )
    role_repo.add_white_painting_entry(role_id, wp)
    return ev


@pytest.fixture()
def trace_recall_service(config: REMSConfig, db, vector_store):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    role = Role(role_id="ROL-trace-1", name="TracerOne")
    role_repo.save(role)
    base = datetime.now() - timedelta(hours=2)
    e_shared = _seed_event_with_wp(
        event_repo=event_repo,
        role_repo=role_repo,
        vector_store=vector_store,
        content="TracerOne 在北京项目讨论数据库优化方案",
        role_id=role.role_id,
        wp_summary="TracerOne 参与技术会议讨论延迟",
        create_time=base,
    )
    e_noise = _seed_event_with_wp(
        event_repo=event_repo,
        role_repo=role_repo,
        vector_store=vector_store,
        content="另一角色在海南度假",
        role_id=role.role_id,
        wp_summary="无关休闲",
        create_time=base + timedelta(hours=1),
    )
    svc = RecallService(config, event_repo, role_repo, vector_store, tri_band=tri_band)
    return svc, event_repo, role_repo, vector_store, role.role_id, e_shared, e_noise


class TestRecallTraceHooks:
    def test_trace_records_tri_band_search(self, trace_recall_service):
        svc, _, _, _, rid, e_shared, e_noise = trace_recall_service
        trace, uninstall = install_recall_trace(svc)
        try:
            focus = {rid}
            ents = [
                EventRoleEntry(
                    role_id=rid,
                    importance=Importance.A,
                    role_snapshot=RoleSnapshot(l2_interaction="讨论延迟与索引"),
                    emotional_model=EmotionalModel(),
                ),
            ]
            block = svc.build_recall_block(
                "数据库优化",
                Shadow(content="上一轮提到服务器"),
                focus_role_ids=focus,
                focus_role_entries=ents,
            )
        finally:
            uninstall()

        assert trace["tri_band_searches"], "tri-band search should run"
        assert block.items, "expected at least one recalled item"
        recalled_ids = {it.event_id for it in block.items}
        assert e_shared.event_id in recalled_ids or e_noise.event_id in recalled_ids

    def test_reinforcement_raises_forgetting_factor(self, trace_recall_service):
        svc, _, role_repo, _, rid, e_shared, _ = trace_recall_service
        wp_before = role_repo.get_white_painting_by_event(rid, e_shared.event_id)
        assert wp_before is not None
        factor_before = wp_before.forgetting_factor

        block = svc.build_recall_block(
            "数据库",
            shadow=None,
            focus_role_ids={rid},
            focus_role_entries=None,
        )
        assert any(it.event_id == e_shared.event_id for it in block.items)

        wp_after = role_repo.get_white_painting_by_event(rid, e_shared.event_id)
        assert wp_after is not None
        assert wp_after.forgetting_factor >= min(
            max(factor_before, 1.0) * svc._config.recall_reinforce_multiplier,
            svc._config.recall_forgetting_factor_cap,
        ) - 1e-6


class TestSummaryTierWithFocus:
    def test_focused_primary_role_gets_more_detailed_summary_level(self, config: REMSConfig, db, vector_store):
        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
        rid_a = "ROL-star"
        rid_b = "ROL-bg"
        role_repo.save(Role(role_id=rid_a, name="主角"))
        role_repo.save(Role(role_id=rid_b, name="配角"))
        entry_star = EventRoleEntry(role_id=rid_a, importance=Importance.S, role_snapshot=RoleSnapshot())
        entry_bg = EventRoleEntry(role_id=rid_b, importance=Importance.D, role_snapshot=RoleSnapshot())
        long_l1 = "x" * 400
        mid_l2 = "y" * 120
        short_l3 = "z" * 40
        ev = Event(
            content_raw="raw",
            summaries={"L1": long_l1, "L2": mid_l2, "L3": short_l3},
            role_list=[entry_bg, entry_star],
        )
        event_repo.save(ev)
        vector_store.upsert_event_vectors(ev)
        wp = WhitePaintingEntry(event_id=ev.event_id, role_summary="白描", forgetting_factor=1.0)
        role_repo.add_white_painting_entry(rid_a, wp)

        svc = RecallService(config, event_repo, role_repo, vector_store, tri_band=tri_band)
        block_focus_star = svc.build_recall_block("anything", focus_role_ids={rid_a})
        block_focus_other = svc.build_recall_block("anything", focus_role_ids={"ROL-nobody"})

        item_star = next(it for it in block_focus_star.items if it.event_id == ev.event_id)
        item_other = next(it for it in block_focus_other.items if it.event_id == ev.event_id)
        assert item_star.summary_level == "L1"
        assert item_other.summary_level == "L3"
