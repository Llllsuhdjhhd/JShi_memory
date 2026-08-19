from __future__ import annotations

from rems.storage.tier1_store import Tier1Store, compute_sample_key


def test_compute_sample_key_monotonic_with_w_eff():
    u = 0.5
    _, s_high = compute_sample_key(10.0, u)
    _, s_low = compute_sample_key(1.0, u)
    assert s_high > s_low


def test_tier1_upsert_orders_by_sample_key(config, db):
    tier1 = Tier1Store(db, config)
    tier1.upsert_tier1("EVT-A", w_i=5.0, asf_i=0.0, sample_u=0.9)
    tier1.upsert_tier1("EVT-B", w_i=50.0, asf_i=0.0, sample_u=0.9)
    tier1.upsert_tier1("EVT-C", w_i=1.0, asf_i=0.0, sample_u=0.01)

    active = tier1.fetch_active_ids(2)
    assert active[0] == "EVT-B"
    assert len(active) == 2


def test_asf_reduces_w_eff(config, db):
    tier1 = Tier1Store(db, config)
    tier1.upsert_tier1("EVT-X", w_i=10.0, asf_i=0.0, sample_u=0.5)
    before = tier1.fetch_active_ids(10)
    tier1.upsert_tier1("EVT-X", w_i=10.0, asf_i=0.5, sample_u=0.5)
    # Higher ASF → lower w_eff → lower sample_key rank when competing
    tier1.upsert_tier1("EVT-Y", w_i=10.0, asf_i=0.0, sample_u=0.5)
    active = tier1.fetch_active_ids(2)
    assert active[0] == "EVT-Y"
