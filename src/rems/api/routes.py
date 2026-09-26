from __future__ import annotations

from typing import Iterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..port import BackendIngestResult, MemoryBatch, MemoryExperience
from .app import get_pipeline

router = APIRouter(tags=["rems"])


# ── Request / Response schemas ────────────────────────────────────────

class IngestRequest(BaseModel):
    subject_id: str
    experiences: list[MemoryExperience] = Field(default_factory=list)


class RecallRequest(BaseModel):
    subject_id: str
    query: str
    object_id: str | None = None
    object_ids: list[str] = Field(default_factory=list)
    interlocutor_object_id: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    budget_chars: int | None = None
    expand_raw: bool = False
    level: int = 1
    limit: int | None = None
    anchor_event_ids: list[str] = Field(default_factory=list)


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


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/ingest", response_model=BackendIngestResult)
def ingest(req: IngestRequest):
    pipeline = get_pipeline()
    return pipeline.ingest_batch(
        MemoryBatch(subject_id=req.subject_id, experiences=tuple(req.experiences))
    )


@router.post("/recall")
def recall(req: RecallRequest):
    pipeline = get_pipeline()
    time_range = None
    if req.time_start and req.time_end:
        from datetime import datetime

        time_range = (
            datetime.fromisoformat(req.time_start),
            datetime.fromisoformat(req.time_end),
        )
    return [
        f.model_dump(mode="json")
        for f in pipeline.recall(
            req.subject_id,
            req.query,
            object_id=req.object_id,
            object_ids=tuple(req.object_ids) or None,
            interlocutor_object_id=req.interlocutor_object_id,
            time_range=time_range,
            budget_chars=req.budget_chars,
            expand_raw=req.expand_raw,
            level=req.level,
            limit=req.limit,
            anchor_event_ids=tuple(req.anchor_event_ids),
        )
    ]


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
