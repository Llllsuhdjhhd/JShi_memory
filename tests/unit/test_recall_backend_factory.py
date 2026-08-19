from __future__ import annotations

from rems.config import REMSConfig
from rems.retrieval import (
    apply_profile_defaults,
    create_recall_backend,
    normalize_recall_profile,
    resolve_recall_profile,
)
from rems.retrieval.backends.hybrid_literary import HybridLiteraryBackend
from rems.retrieval.backends.tri_band import TriBandOnlyBackend
from rems.retrieval.bm25_index import EventBm25Index
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository


def test_normalize_profile_aliases():
    assert normalize_recall_profile("hybrid") == "hybrid_literary"
    assert normalize_recall_profile("passive_log") == "personal_log"


def test_resolve_profile_honors_explicit_override():
    cfg = REMSConfig(recall_profile="hybrid_literary")
    assert resolve_recall_profile(cfg, mode="dialogue") == "hybrid_literary"


def test_resolve_profile_uses_mode_when_default_tri_band():
    cfg = REMSConfig(recall_profile="tri_band")
    assert resolve_recall_profile(cfg, mode="passive_log") == "personal_log"


def test_apply_hybrid_defaults(tmp_path):
    cfg = REMSConfig()
    apply_profile_defaults(cfg, "hybrid_literary")
    assert cfg.recall_bm25_enabled is True
    assert cfg.recall_recency_enabled is True
    assert "bm25" in cfg.recall_rrf_channels


def test_create_backend_types(tmp_path):
    cfg = REMSConfig(recall_profile="tri_band")
    db = Database(f"sqlite:///{tmp_path / 'f.db'}")
    db.create_tables()
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    bm25 = EventBm25Index()
    assert isinstance(
        create_recall_backend(cfg, event_repo, role_repo, bm25),
        TriBandOnlyBackend,
    )
    cfg.recall_profile = "hybrid_literary"
    apply_profile_defaults(cfg, "hybrid_literary")
    assert isinstance(
        create_recall_backend(cfg, event_repo, role_repo, bm25),
        HybridLiteraryBackend,
    )
