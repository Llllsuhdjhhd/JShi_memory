from __future__ import annotations

from ..config import REMSConfig
from ..storage.repository import EventRepository, RoleRepository
from .backends.hybrid_literary import HybridLiteraryBackend
from .backends.npc_agent import NpcAgentBackend
from .backends.personal_log import PersonalLogBackend
from .backends.tri_band import TriBandOnlyBackend
from .bm25_index import EventBm25Index
from .protocol import RecallBackend

_PROFILE_ALIASES = {
    "tri_band": "tri_band",
    "default": "tri_band",
    "hybrid_literary": "hybrid_literary",
    "hybrid": "hybrid_literary",
    "literary": "hybrid_literary",
    "personal_log": "personal_log",
    "passive_log": "personal_log",
    "npc_agent": "npc_agent",
    "npc": "npc_agent",
}

_MODE_DEFAULT_PROFILE: dict[str, str] = {
    "dialogue": "tri_band",
    "passive_log": "personal_log",
    "npc_agent": "npc_agent",
}


def normalize_recall_profile(name: str | None) -> str:
    if not name:
        return "tri_band"
    return _PROFILE_ALIASES.get(name.strip().lower(), name.strip().lower())


def resolve_recall_profile(config: REMSConfig, mode: str | None = None) -> str:
    raw = (getattr(config, "recall_profile", None) or "").strip().lower()
    if raw and raw not in ("tri_band", "default", ""):
        return normalize_recall_profile(raw)
    if mode is not None:
        key = mode.value if hasattr(mode, "value") else str(mode)
        return _MODE_DEFAULT_PROFILE.get(key.lower(), "tri_band")
    return "tri_band"


def create_recall_backend(
    config: REMSConfig,
    event_repo: EventRepository,
    role_repo: RoleRepository,
    bm25_index: EventBm25Index,
    *,
    mode: str | None = None,
) -> RecallBackend:
    profile = resolve_recall_profile(config, mode)
    if profile == "hybrid_literary":
        return HybridLiteraryBackend(config, event_repo, bm25_index)
    if profile == "personal_log":
        return PersonalLogBackend(config, event_repo, bm25_index)
    if profile == "npc_agent":
        return NpcAgentBackend(config, event_repo, role_repo, bm25_index)
    return TriBandOnlyBackend(config)


def apply_profile_defaults(config: REMSConfig, profile: str) -> None:
    """Set channel flags when profile is selected without explicit overrides."""
    p = normalize_recall_profile(profile)
    if p == "tri_band":
        return
    if p == "hybrid_literary":
        if not config.recall_rrf_channels:
            config.recall_rrf_channels = ["tri_band", "bm25", "recency"]
        config.recall_bm25_enabled = True
        config.recall_recency_enabled = True
        config.recall_enrichment_metadata_enabled = True
    elif p == "personal_log":
        if not config.recall_rrf_channels:
            config.recall_rrf_channels = ["tri_band", "recency"]
        config.recall_recency_enabled = True
        config.recall_bm25_enabled = getattr(config, "recall_bm25_enabled", False)
    elif p == "npc_agent":
        if not config.recall_rrf_channels:
            config.recall_rrf_channels = ["tri_band", "role_anchor", "recency"]
        config.recall_recency_enabled = True
