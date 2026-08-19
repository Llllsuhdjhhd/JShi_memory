from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rems.config import REMSConfig
from rems.llm.prompt_registry import PromptRegistry, PromptTemplate
from rems.models.event import Event, EventRoleEntry, Importance
from rems.models.role import WhitePaintingEntry
from rems.strategies.forgetting import DefaultWhitePaintingRetentionStrategy
from rems.strategies.recall import DefaultRecallScoringStrategy, DefaultSummaryTierPolicy


def test_default_recall_scoring_exposes_breakdown():
    config = REMSConfig()
    strategy = DefaultRecallScoringStrategy(config)
    event = Event(
        content_raw="张三推进项目。",
        role_list=[EventRoleEntry(role_id="ROL-1", importance=Importance.A)],
        activation_energy=0.2,
    )

    breakdown = strategy.score(event, cosine_sim=0.7)

    assert breakdown.cosine == 0.7
    assert breakdown.role_boost == 0.2
    assert 0.0 <= breakdown.total <= 1.0


def test_default_summary_tier_policy_is_role_aware():
    config = REMSConfig()
    policy = DefaultSummaryTierPolicy(config)
    event = Event(
        content_raw="张三推进项目。",
        role_list=[EventRoleEntry(role_id="ROL-1", importance=Importance.A)],
    )

    focused = policy.tier_offset(event, {"ROL-1"})
    unfocused = policy.tier_offset(event, {"ROL-other"})

    assert focused < unfocused


@pytest.mark.parametrize(
    "importance,focus,expected_key",
    [
        (Importance.S, {"ROL-x"}, "sa"),
        (Importance.A, {"ROL-x"}, "sa"),
        (Importance.B, {"ROL-x"}, "b"),
        (Importance.C, {"ROL-x"}, "cd"),
        (Importance.D, {"ROL-x"}, "cd"),
        (Importance.A, set(), "empty_focus"),
        (Importance.A, {"ROL-other"}, "no_focus_match"),
    ],
)
def test_summary_tier_offset_matrix(importance, focus, expected_key):
    """Relative tier offsets: S/A < B < C/D / absent when focus applies (recall Lazy Index)."""
    cfg = REMSConfig()
    policy = DefaultSummaryTierPolicy(cfg)
    ev = Event(
        content_raw="x",
        role_list=[EventRoleEntry(role_id="ROL-x", importance=importance)],
    )
    off = policy.tier_offset(ev, focus)

    mid_default = cfg.recall_default_tier_offset
    shift_p = cfg.recall_primary_role_detail_shift
    shift_m = cfg.recall_minor_role_compress_shift

    if expected_key == "sa":
        assert off == mid_default - shift_p
    elif expected_key == "b":
        assert off == mid_default
    elif expected_key == "cd":
        assert off == mid_default + shift_m
    elif expected_key == "empty_focus":
        assert off == mid_default
    elif expected_key == "no_focus_match":
        assert off == mid_default + shift_m


def test_default_forgetting_strategy_penalizes_old_entries():
    config = REMSConfig()
    strategy = DefaultWhitePaintingRetentionStrategy(config)
    entry = WhitePaintingEntry(
        event_id="EVT-1",
        role_summary="旧白描",
        create_time=datetime.now() - timedelta(days=120),
        memory_weight=0.1,
    )

    unpenalized = strategy.score(entry, is_penalized=False)
    penalized = strategy.score(entry, is_penalized=True, now=datetime.now())

    assert unpenalized.score == 1.0
    assert penalized.score < 1.0
    assert penalized.is_penalized is True


def test_prompt_registry_supports_versions_and_defaults():
    registry = PromptRegistry()
    v1 = PromptTemplate(task_type="abstraction", version="v1", system="s1", user="u1")
    v2 = PromptTemplate(task_type="abstraction", version="v2", system="s2", user="u2")

    registry.register(v1, make_default=True)
    registry.register(v2)

    assert registry.get("abstraction").version == "v1"
    assert registry.get("abstraction", "v2").system == "s2"
    assert registry.default_version("abstraction") == "v1"
