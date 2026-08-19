"""Unit tests for PerfMonitor: rolling stats, dual-channel load factors."""

from __future__ import annotations

from rems.observability import PerfMonitor


def test_disabled_monitor_is_inert():
    pm = PerfMonitor(enabled=False, tolerances_ms={"x": 1.0})
    with pm.timer("x"):
        pass
    assert pm.load_factor() == 1.0
    assert pm.load_factor_recall() == 1.0
    assert pm.load_factor_abstract() == 1.0
    assert pm.is_overloaded() is False
    assert pm.adjusted_silence_threshold(0.02) == 0.02
    assert pm.adjusted_recent_k(200, 20) == 200
    assert pm.adjusted_active_pool_limit(100_000, 20_000) == 100_000


def test_record_and_mean():
    pm = PerfMonitor(tolerances_ms={"phase_a": 100.0})
    for v in [10.0, 20.0, 30.0]:
        pm.record("phase_a", v)
    assert pm.mean_ms("phase_a") == 20.0


def test_rolling_window_evicts_oldest():
    pm = PerfMonitor(window_size=3, tolerances_ms={"p": 100.0})
    for v in [100.0, 200.0, 300.0, 400.0]:
        pm.record("p", v)
    assert pm.mean_ms("p") == 300.0


def test_load_factor_recall_only_uses_recall_phases():
    pm = PerfMonitor(
        tolerances_ms={"rag_search": 100.0, "abstraction_mining": 100.0},
        load_factor_max=4.0,
        recall_phases=["rag_search"],
        abstract_phases=["abstraction_mining"],
    )
    pm.record("abstraction_mining", 500.0)
    assert pm.load_factor_recall() == 1.0
    assert pm.load_factor_abstract() == 4.0


def test_adjusted_active_pool_limit_shrinks_with_recall_load():
    pm = PerfMonitor(
        tolerances_ms={"rag_search": 100.0},
        load_factor_max=4.0,
        recall_phases=["rag_search"],
    )
    assert pm.adjusted_active_pool_limit(100_000, 20_000) == 100_000
    pm.record("rag_search", 1000.0)
    assert pm.adjusted_active_pool_limit(100_000, 20_000) == 20_000


def test_adjusted_recent_k_uses_abstract_channel():
    pm = PerfMonitor(
        tolerances_ms={"abstraction_mining": 100.0},
        load_factor_max=4.0,
        abstract_phases=["abstraction_mining"],
    )
    assert pm.adjusted_recent_k(200, 20) == 200
    pm.record("abstraction_mining", 1000.0)
    assert pm.adjusted_recent_k(200, 20) == 20


def test_adjusted_silence_threshold_deprecated_returns_base():
    pm = PerfMonitor(
        tolerances_ms={"p": 100.0},
        load_factor_max=4.0,
        silence_boost_at_max=4.0,
    )
    pm.record("p", 1000.0)
    assert pm.adjusted_silence_threshold(0.02) == 0.02


def test_timer_context_records_elapsed():
    import time

    pm = PerfMonitor(tolerances_ms={"sleep_phase": 1000.0})
    with pm.timer("sleep_phase"):
        time.sleep(0.01)
    samples = pm.stats_snapshot()["sleep_phase"].samples
    assert len(samples) == 1
    assert 5.0 < samples[0] < 1000.0
