from __future__ import annotations

# HTTP API：将白皮书中的 ingest、回忆侧数据、墓碑、角色双轨、后台演化等暴露为 REST。
# 注意：ingest 返回 ID 列表；完整 ContextPackage 由管线内部生成，如需返回需扩展响应模型。

from contextlib import contextmanager
from typing import Iterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import UserMode
from ..pipeline import ProcessingMode
from .app import get_pipeline

router = APIRouter(tags=["rems"])


# ── Request / Response schemas ────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str
    force_save: bool = False
    mode: ProcessingMode = ProcessingMode.DIALOGUE
    npc_role_id: str | None = None

    # Per-request overrides for 单人/多人模式（白皮书 2.2）。若留空则沿用全局 REMSConfig。
    user_mode: UserMode | None = None
    core_user_role_id: str | None = None
    active_participants: list[str] | None = None


class IngestResponse(BaseModel):
    sealed_event_ids: list[str] = Field(default_factory=list)
    abstract_event_ids: list[str] = Field(default_factory=list)
    mode: str = "dialogue"
    npc_directives: list[dict] = Field(default_factory=list)


class RoleQuery(BaseModel):
    name_or_id: str


class TombstoneRequest(BaseModel):
    event_id: str
    reason: str
    replacement_event_id: str | None = None


class EventOut(BaseModel):
    event_id: str
    content_raw: str
    summaries: dict[str, str]
    is_abstract: bool
    is_tombstoned: bool = False
    status: str
    insight: str | None = None
    affective_energy: float = 0.0
    activation_energy: float = 0.0
    abstract_coverage: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────

@contextmanager
def _mode_override(req: IngestRequest) -> Iterator[None]:
    """Temporarily patch pipeline.config with per-request mode overrides.

    单次调用内覆盖 ``user_mode`` / ``core_user_role_id`` / ``active_participants``；
    上下文退出后恢复原值（进程安全；非并发-请求安全，若需高并发请求级隔离，
    应改为在 pipeline/skill 签名中显式传 ctx 参数而不是 mutate config）。
    """
    pipeline = get_pipeline()
    cfg = pipeline.config
    originals: dict[str, object] = {}
    try:
        if req.user_mode is not None:
            originals["user_mode"] = cfg.user_mode
            cfg.user_mode = req.user_mode
        if req.core_user_role_id is not None:
            originals["core_user_role_id"] = cfg.core_user_role_id
            cfg.core_user_role_id = req.core_user_role_id
        if req.active_participants is not None:
            originals["active_participants"] = list(cfg.active_participants)
            cfg.active_participants = req.active_participants
        yield
    finally:
        for k, v in originals.items():
            setattr(cfg, k, v)


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    pipeline = get_pipeline()
    with _mode_override(req):
        result = pipeline.ingest(
            req.text,
            force_save=req.force_save,
            mode=req.mode,
            npc_role_id=req.npc_role_id,
        )
    return IngestResponse(
        sealed_event_ids=[e.event_id for e in result.sealed_events],
        abstract_event_ids=[e.event_id for e in result.abstract_events],
        mode=result.mode.value,
        npc_directives=result.npc_directives,
    )


@router.get("/events", response_model=list[EventOut])
def list_events(is_abstract: bool | None = None, include_tombstoned: bool = False):
    pipeline = get_pipeline()
    events = pipeline.event_repo.list_all(
        is_abstract=is_abstract,
        exclude_tombstoned=not include_tombstoned,
    )
    return [_event_out(e) for e in events]


@router.get("/events/{event_id}", response_model=EventOut)
def get_event(event_id: str):
    pipeline = get_pipeline()
    e = pipeline.event_service.get_event(event_id)
    if not e:
        raise HTTPException(404, f"Event {event_id} not found")
    return _event_out(e)


@router.post("/events/tombstone")
def tombstone_event(req: TombstoneRequest):
    pipeline = get_pipeline()
    ok = pipeline.tombstone(req.event_id, req.reason, req.replacement_event_id)
    if not ok:
        raise HTTPException(404, f"Event {req.event_id} not found")
    return {"tombstoned": req.event_id}


@router.post("/roles/query")
def query_role(req: RoleQuery):
    pipeline = get_pipeline()
    result = pipeline.query_role(req.name_or_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@router.get("/roles")
def list_roles():
    pipeline = get_pipeline()
    roles = pipeline.role_service.list_roles()
    return [r.model_dump(mode="json", exclude={"white_painting"}) for r in roles]


@router.get("/roles/{role_id}/conflicts")
def detect_conflicts(role_id: str, top_n: int = 50):
    pipeline = get_pipeline()
    conflicts = pipeline.belief_revision_service.detect_conflicts(role_id, top_n=top_n)
    return {"role_id": role_id, "conflicts": conflicts}


@router.post("/evolve")
def evolve():
    pipeline = get_pipeline()
    new_abstracts = pipeline.run_evolution()
    return {"created": [e.event_id for e in new_abstracts]}


@router.get("/health")
def health():
    return {"status": "ok"}


# ── Serialisation helpers ────────────────────────────────────────────

def _event_out(e) -> EventOut:
    return EventOut(
        event_id=e.event_id,
        content_raw=e.content_raw,
        summaries=e.summaries,
        is_abstract=e.is_abstract,
        is_tombstoned=e.is_tombstoned,
        status=e.status.value,
        insight=e.insight,
        affective_energy=round(e.affective_energy, 4),
        activation_energy=round(e.activation_energy, 4),
        abstract_coverage=round(float(getattr(e, "abstract_coverage", 0.0) or 0.0), 4),
    )
