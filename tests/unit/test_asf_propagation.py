from __future__ import annotations

from unittest.mock import MagicMock

from rems.config import REMSConfig
from rems.models.event import Event
from rems.services.abstraction_shadowing import AbstractionShadowingService
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.tier1_store import Tier1Store
from rems.storage.vector_store import VectorStore
from rems.embedding.tri_band import TriBandEncoder


def test_asf_propagation_multiplies_survival(config, db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    tier1 = Tier1Store(db, config)
    vector = VectorStore(config)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    vector.set_tri_band(tri_band)

    leaf = Event(content_raw="基本事件内容较长一些用于长度比")
    abstract = Event(content_raw="抽象", is_abstract=True, source_events=[leaf.event_id])
    event_repo.save(leaf)
    event_repo.save(abstract)
    vector.upsert_event_vectors(leaf)
    vector.upsert_event_vectors(abstract)
    tier1.upsert_tier1(leaf.event_id, w_i=10.0, asf_i=0.0)

    svc = AbstractionShadowingService(config, event_repo, role_repo, vector, tier1)
    svc._vector.cosine_between_events = MagicMock(return_value=0.8)  # type: ignore[method-assign]

    svc.propagate(abstract, shadow_lambda=0.5)
    asf = tier1.get_asf(leaf.event_id)
    hop = 0.8 * min(1.0, len(abstract.content_raw) / len(leaf.content_raw))
    incoming = hop * 0.5
    expected = 1.0 - (1.0 - 0.0) * (1.0 - incoming)
    assert abs(asf - expected) < 1e-5


def test_ptsd_immune_skips_asf(config, db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    tier1 = Tier1Store(db, config)
    vector = VectorStore(config)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    vector.set_tri_band(tri_band)

    leaf = Event(content_raw="免疫事件", ptsd_immune=True)
    abstract = Event(content_raw="抽象覆盖", is_abstract=True, source_events=[leaf.event_id])
    event_repo.save(leaf)
    event_repo.save(abstract)
    tier1.upsert_tier1(leaf.event_id, w_i=10.0, asf_i=0.0)

    svc = AbstractionShadowingService(config, event_repo, role_repo, vector, tier1)
    svc._vector.cosine_between_events = MagicMock(return_value=0.9)  # type: ignore[method-assign]
    svc.propagate(abstract, shadow_lambda=1.0)
    assert tier1.get_asf(leaf.event_id) == 0.0
