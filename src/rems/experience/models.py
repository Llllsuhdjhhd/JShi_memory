"""Contracts for two independent forms of long term experience."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


ExperienceStatus = Literal["established", "tentative"]
PortraitRevisionKind = Literal["initial", "confirm", "extend", "revise", "uncertain"]


@dataclass(frozen=True)
class ContextualAdjustment:
    context: str
    approach: str


@dataclass(frozen=True)
class RevisedClaim:
    previous: str
    updated: str
    reason: str


@dataclass(frozen=True)
class ExperienceCandidate:
    """A structured general insight proposed by a future Dream implementation."""

    core: str
    evidence_event_ids: tuple[str, ...]
    applicable_context: str | None = None
    action_or_expectation: str | None = None
    observed_outcome: str | None = None
    exceptions: tuple[str, ...] = ()
    status: ExperienceStatus = "tentative"


@dataclass(frozen=True)
class GeneralExperience:
    experience_id: str
    subject_id: str
    core: str
    evidence_event_ids: tuple[str, ...]
    applicable_context: str | None
    action_or_expectation: str | None
    observed_outcome: str | None
    exceptions: tuple[str, ...]
    status: ExperienceStatus
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class ExperiencePortraitSyncResult:
    object_id: str
    imported_white_paintings: int
    updated: bool
    pending_white_paintings: int = 0
    pending_chars: int = 0
    review_outcome: str = "waiting"


@dataclass(frozen=True)
class PersonExperiencePortrait:
    """A concise interaction stance formed from, but distinct from, white paintings."""

    subject_id: str
    object_id: str
    content: str
    brief: str
    baseline: str
    adjustments: tuple[ContextualAdjustment, ...]
    uncertainties: tuple[str, ...]
    version: int
    source_chars: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PersonExperienceRevision:
    revision_id: str
    subject_id: str
    object_id: str
    version: int
    kind: PortraitRevisionKind
    reason: str
    revised_claims: tuple[RevisedClaim, ...]
    evidence_event_ids: tuple[str, ...]
    content: str
    brief: str
    baseline: str
    adjustments: tuple[ContextualAdjustment, ...]
    uncertainties: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True)
class PersonExperienceFragment:
    object_id: str
    content: str
    variant: str
    source_event_ids: tuple[str, ...]
    updated_at: datetime
    score: float = 0.0
    type: str = "person_experience"


@dataclass(frozen=True)
class GeneralExperienceFragment:
    experience_id: str
    content: str
    evidence_event_ids: tuple[str, ...]
    applicable_context: str | None
    action_or_expectation: str | None
    observed_outcome: str | None
    exceptions: tuple[str, ...]
    score: float
    type: str = "general_experience"


@dataclass(frozen=True)
class ExperienceRecallQuery:
    subject_id: str
    query: str
    object_ids: tuple[str, ...] = ()
    budget_chars: int = 2000
    person_limit: int = 3
    general_limit: int = 5


@dataclass(frozen=True)
class ExperienceRecallResult:
    person: tuple[PersonExperienceFragment, ...] = ()
    general: tuple[GeneralExperienceFragment, ...] = ()

    @property
    def used_chars(self) -> int:
        return sum(len(item.content) for item in (*self.person, *self.general))
