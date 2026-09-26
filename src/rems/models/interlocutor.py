from __future__ import annotations

from pydantic import BaseModel, Field


class InterlocutorAttribution(BaseModel):
    """A historical interlocutor tied to the experience segment that supplied it."""

    segment_id: str = Field(..., description="Source experience segment ID")
    object_id: str = Field(..., description="JShi object ID for the interlocutor")
