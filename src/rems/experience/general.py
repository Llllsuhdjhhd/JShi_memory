"""Validation boundary for future Dream experience proposals."""

from __future__ import annotations

from uuid import uuid4

from ..models.event import EventStatus
from .models import ExperienceCandidate, GeneralExperience
from .ports import EventEvidencePort, GeneralExperienceStorePort


class DreamCandidateSink:
    def __init__(
        self, events: EventEvidencePort, store: GeneralExperienceStorePort
    ) -> None:
        self._events = events
        self._store = store

    def submit(
        self, subject_id: str, candidates: tuple[ExperienceCandidate, ...]
    ) -> tuple[str, ...]:
        """Accept proposed insights after checking their original Event evidence."""
        if not subject_id.strip():
            raise ValueError("subject_id is required")
        ready: list[GeneralExperience] = []
        for candidate in candidates:
            if (
                not isinstance(candidate.core, str)
                or any(not isinstance(value, str) for value in (
                    candidate.applicable_context,
                    candidate.action_or_expectation,
                    candidate.observed_outcome,
                ) if value is not None)
                or any(not isinstance(item, str) for item in candidate.exceptions)
            ):
                raise ValueError("invalid experience fields")
            core = candidate.core.strip()
            event_ids = tuple(dict.fromkeys(candidate.evidence_event_ids))
            context = (candidate.applicable_context or "").strip() or None
            outcome = (candidate.observed_outcome or "").strip() or None
            action = (candidate.action_or_expectation or "").strip() or None
            exceptions = tuple(dict.fromkeys(
                item.strip() for item in candidate.exceptions if item.strip()
            ))
            if not core or not event_ids:
                raise ValueError("candidate needs a core insight and Event evidence")
            if candidate.status not in {"established", "tentative"}:
                raise ValueError("invalid experience status")
            if candidate.status == "established" and (not context or not outcome):
                raise ValueError("established experience needs context and observed outcome")
            for event_id in event_ids:
                event = self._events.get(event_id)
                if (
                    event is None
                    or event.subject_id != subject_id
                    or event.is_tombstoned
                    or event.is_abstract
                    or event.status == EventStatus.UNCLOSED
                ):
                    raise ValueError(f"invalid evidence Event: {event_id}")
            item = GeneralExperience(
                experience_id=f"EXP-{uuid4().hex}",
                subject_id=subject_id,
                core=core,
                evidence_event_ids=event_ids,
                applicable_context=context,
                action_or_expectation=action,
                observed_outcome=outcome,
                exceptions=exceptions,
                status=candidate.status,
            )
            ready.append(item)
        for item in ready:
            self._store.save(item)
        return tuple(item.experience_id for item in ready)
