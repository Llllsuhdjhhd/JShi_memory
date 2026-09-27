"""Experience framework: portrait experience and a future Dream input boundary."""

from .general import DreamCandidateSink
from .models import (
    ContextualAdjustment,
    ExperienceCandidate,
    ExperienceRecallQuery,
    ExperienceRecallResult,
    GeneralExperience,
    GeneralExperienceFragment,
    PersonExperienceFragment,
    PersonExperiencePortrait,
    PersonExperienceRevision,
    RevisedClaim,
    ExperiencePortraitSyncResult,
)
from .module import ExperienceModule
from .person import ExperiencePortraitReader, WhitePaintingExperiencePortraitBuilder
from .person_composer import (
    InsufficientEvidence,
    InteractionPortraitDraft,
    LLMPersonExperienceComposer,
    PersonExperienceComposerPort,
)
from .store import PersonExperienceRepository

__all__ = [
    "ContextualAdjustment",
    "DreamCandidateSink",
    "ExperienceCandidate",
    "ExperienceModule",
    "ExperienceRecallQuery",
    "ExperienceRecallResult",
    "GeneralExperience",
    "GeneralExperienceFragment",
    "PersonExperienceFragment",
    "PersonExperiencePortrait",
    "PersonExperienceRevision",
    "RevisedClaim",
    "ExperiencePortraitReader",
    "ExperiencePortraitSyncResult",
    "WhitePaintingExperiencePortraitBuilder",
    "InteractionPortraitDraft",
    "InsufficientEvidence",
    "LLMPersonExperienceComposer",
    "PersonExperienceComposerPort",
    "PersonExperienceRepository",
]
