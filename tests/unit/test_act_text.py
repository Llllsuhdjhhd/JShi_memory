from __future__ import annotations

from rems.config import REMSConfig, TriBandConfig
from rems.embedding.act_text import (
    clamp_embed_text,
    pick_summary,
    query_act_compress_bounds,
    resolve_event_act_text,
    resolve_query_act_text,
)
from rems.models.event import Event


def _tb(**kwargs) -> TriBandConfig:
    base = REMSConfig().tri_band
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


def test_pick_summary_returns_none_when_missing():
    ev = Event(content_raw="raw")
    assert pick_summary(ev, "L1") is None


def test_resolve_event_act_text_short_uses_l1():
    tb = _tb(act_short_raw_chars=800, act_short_level="L1")
    ev = Event(content_raw="short", summaries={"L1": "L1 summary"})
    assert resolve_event_act_text(ev, tb) == "L1 summary"


def test_resolve_event_act_text_short_fallback_raw():
    tb = _tb(act_short_raw_chars=800)
    ev = Event(content_raw="short only")
    assert resolve_event_act_text(ev, tb) == "short only"


def test_resolve_event_act_text_long_uses_l2():
    tb = _tb(act_short_raw_chars=10, act_long_level="L2", act_embed_max_chars=480)
    raw = "x" * 20
    ev = Event(content_raw=raw, summaries={"L2": "L2 summary", "L1": "L1 summary"})
    assert resolve_event_act_text(ev, tb) == "L2 summary"


def test_resolve_event_act_text_long_fallback_l1_then_raw():
    tb = _tb(act_short_raw_chars=5, act_long_level="L2", act_embed_max_chars=480)
    raw = "abcdefghij"
    ev = Event(content_raw=raw, summaries={"L1": "L1 only"})
    assert resolve_event_act_text(ev, tb) == "L1 only"


def test_resolve_event_act_text_picks_longest_fitting_summary():
    tb = _tb(act_short_raw_chars=5, act_long_level="L2", act_embed_max_chars=40)
    ev = Event(
        content_raw="x" * 100,
        summaries={
            "L2": "L2 " * 20,
            "L3": "L3短摘要",
            "L1": "L1中等长度摘要内容",
        },
    )
    assert resolve_event_act_text(ev, tb) == "L1中等长度摘要内容"


def test_resolve_event_act_text_clamps_when_all_too_long():
    tb = _tb(act_short_raw_chars=9999, act_embed_max_chars=5)
    ev = Event(content_raw="1234567890", summaries={"L1": "1234567890"})
    assert resolve_event_act_text(ev, tb) == "12345"


def test_clamp_embed_text():
    assert clamp_embed_text("abcdef", 10) == "abcdef"
    assert clamp_embed_text("abcdefgh", 5) == "abcde"


def test_resolve_query_act_text_no_truncation():
    tb = _tb(query_act_max_chars=120)
    assert resolve_query_act_text("hello", tb) == "hello"


def test_resolve_query_act_text_tail():
    tb = _tb(query_act_max_chars=5, query_act_use_tail=True)
    assert resolve_query_act_text("abcdefgh", tb) == "defgh"


def test_resolve_query_act_text_head():
    tb = _tb(query_act_max_chars=5, query_act_use_tail=False)
    assert resolve_query_act_text("abcdefgh", tb) == "abcde"


def test_query_act_compress_bounds():
    tb = _tb(query_act_max_chars=200, query_act_compress_max_chars=400)
    cap, target, min_chars = query_act_compress_bounds(875, tb)
    assert cap == 400
    assert target == 200
    assert min_chars == 200
    _, _, short_min = query_act_compress_bounds(40, tb)
    assert short_min == 40
    _, _, passthrough = query_act_compress_bounds(150, tb)
    assert passthrough == 150
