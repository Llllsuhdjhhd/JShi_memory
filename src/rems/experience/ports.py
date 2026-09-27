"""Narrow boundaries; Dream and durable general indexing remain separate."""

from __future__ import annotations

from typing import Protocol

from ..models.event import Event
from .models import GeneralExperience


class EventEvidencePort(Protocol):
    def get(self, event_id: str) -> Event | None: ...


class GeneralExperienceStorePort(Protocol):
    def save(self, experience: GeneralExperience) -> None: ...


class GeneralExperienceSearchPort(Protocol):
    """Search must be subject scoped and return a bounded candidate set."""

    def search(
        self, subject_id: str, query: str, *, limit: int
    ) -> tuple[tuple[GeneralExperience, float], ...]: ...
