from __future__ import annotations

from rems.config import REMSConfig
from rems.embedding.tri_band import TriBandEncoder
from rems.models.event import Event, EventRoleEntry, Importance
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.models.role import Role


def test_abstract_ent_uses_leaf_roles(config, db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    role_repo.save(Role(role_id="ROL-a", name="Alice", aliases=["阿丽"]))
    basic = Event(
        content_raw="Alice met Bob",
        role_list=[EventRoleEntry(role_id="ROL-a", importance=Importance.A)],
    )
    event_repo.save(basic)
    abstract = Event(
        content_raw="pattern",
        is_abstract=True,
        source_events=[basic.event_id],
        role_list=[],
    )
    event_repo.save(abstract)

    enc = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    ent_text = enc._entity_text(abstract)
    assert "ROL-a" not in ent_text
    assert "Alice" in ent_text
    assert "阿丽" in ent_text
    vecs = enc.encode(abstract)
    assert len(vecs.vector_ent) == config.tri_band.vector_dim_ent


def test_basic_event_ent_from_role_list(config, db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    role_repo.save(Role(role_id="ROL-b", name="Bob"))
    ev = Event(
        content_raw="Bob spoke",
        role_list=[EventRoleEntry(role_id="ROL-b", importance=Importance.B)],
    )
    enc = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    text = enc._entity_text(ev)
    assert "Bob" in text
    assert "ROL-b" not in text


def test_act_text_adaptive_short_vs_long(config):
    enc = TriBandEncoder(config)
    short = Event(
        content_raw="short",
        summaries={"L1": "short L1", "L2": "short L2"},
    )
    long_raw = "x" * (config.tri_band.act_short_raw_chars + 1)
    long_ev = Event(
        content_raw=long_raw,
        summaries={"L1": "long L1", "L2": "long L2"},
    )
    assert enc._act_text(short) == "short L1"
    assert enc._act_text(long_ev) == "long L2"


def test_encode_query_act_truncation(config):
    enc = TriBandEncoder(config)
    long_q = "a" * 200
    cap = config.tri_band.query_act_max_chars
    vecs_full = enc.encode_query(long_q)
    tail = long_q[-cap:]
    vecs_tail = enc.encode_query(long_q, act_source=tail)
    assert vecs_full.vector_act == vecs_tail.vector_act
