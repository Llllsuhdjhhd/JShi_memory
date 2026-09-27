"""Orchestration for person and general experience recall."""

from __future__ import annotations

from itertools import islice
from typing import Callable

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..storage.repository import EventRepository, RoleRepository
from .general import DreamCandidateSink
from .models import (
    ExperienceCandidate,
    ExperienceRecallQuery,
    ExperienceRecallResult,
    GeneralExperienceFragment,
    ExperiencePortraitSyncResult,
)
from .person import ExperiencePortraitReader, WhitePaintingExperiencePortraitBuilder
from .person_composer import LLMPersonExperienceComposer, PersonExperienceComposerPort
from .ports import GeneralExperienceSearchPort, GeneralExperienceStorePort
from .store import PersonExperienceRepository


class ExperienceModule:
    """Independent experience entry points; no change to Event ingestion/recall."""

    def __init__(
        self,
        person_builder: WhitePaintingExperiencePortraitBuilder,
        person_reader: ExperiencePortraitReader,
        *,
        general_search: GeneralExperienceSearchPort | None = None,
        dream_sink: DreamCandidateSink | None = None,
    ) -> None:
        self._person_builder = person_builder
        self._person_reader = person_reader
        self._general_search = general_search
        self._dream_sink = dream_sink

    @classmethod
    def from_repositories(
        cls,
        config: REMSConfig,
        white_paintings: RoleRepository,
        portraits: PersonExperienceRepository,
        *,
        llm: LLMProvider | None = None,
        person_composer: PersonExperienceComposerPort | None = None,
        person_relevance: Callable[[str, str, str], float] | None = None,
        events: EventRepository | None = None,
        general_store: GeneralExperienceStorePort | None = None,
        general_search: GeneralExperienceSearchPort | None = None,
    ) -> ExperienceModule:
        """Assemble the framework without wiring it into the existing pipeline."""
        composer = person_composer or (
            LLMPersonExperienceComposer(llm) if llm is not None else None
        )
        builder = WhitePaintingExperiencePortraitBuilder(
            config, white_paintings, portraits, composer
        )
        reader = ExperiencePortraitReader(portraits, relevance=person_relevance)
        sink = (
            DreamCandidateSink(events, general_store)
            if events is not None and general_store is not None
            else None
        )
        return cls(builder, reader, general_search=general_search, dream_sink=sink)

    def refresh_person(self, subject_id: str, object_id: str) -> ExperiencePortraitSyncResult:
        """Import any new white paintings and update this person's portrait."""
        return self._person_builder.sync(subject_id, object_id)

    def submit_dream_candidates(
        self, subject_id: str, candidates: tuple[ExperienceCandidate, ...]
    ) -> tuple[str, ...]:
        """Future Dream integration point. Extraction itself lives in Dream."""
        if self._dream_sink is None:
            raise RuntimeError("general experience storage is not configured")
        return self._dream_sink.submit(subject_id, candidates)

    def recall(self, request: ExperienceRecallQuery) -> ExperienceRecallResult:
        if not request.subject_id.strip():
            raise ValueError("subject_id is required")
        if request.budget_chars < 0 or request.person_limit < 0 or request.general_limit < 0:
            raise ValueError("budgets and limits must not be negative")
        if request.budget_chars == 0:
            return ExperienceRecallResult()

        general_hits = ()
        if (
            self._general_search is not None
            and request.query.strip()
            and request.general_limit > 0
        ):
            # The search provider owns the bounded index and candidate ranking.
            general_hits = self._general_search.search(
                request.subject_id,
                request.query,
                limit=request.general_limit,
            )

        person_budget = (
            request.budget_chars // 2 if general_hits else request.budget_chars
        )
        person = self._person_reader.recall(
            request.subject_id,
            request.object_ids,
            query=request.query,
            budget_chars=person_budget,
            limit=request.person_limit,
        )
        remaining = request.budget_chars - sum(len(item.content) for item in person)

        general: list[GeneralExperienceFragment] = []
        for experience, score in islice(general_hits, request.general_limit):
            if experience.subject_id != request.subject_id or experience.status != "established":
                continue
            content = experience.core.strip()
            if not content or len(content) > remaining:
                continue
            general.append(GeneralExperienceFragment(
                experience_id=experience.experience_id,
                content=content,
                evidence_event_ids=experience.evidence_event_ids,
                applicable_context=experience.applicable_context,
                action_or_expectation=experience.action_or_expectation,
                observed_outcome=experience.observed_outcome,
                exceptions=experience.exceptions,
                score=score,
            ))
            remaining -= len(content)

        if not general and person_budget != request.budget_chars:
            person = self._person_reader.recall(
                request.subject_id,
                request.object_ids,
                query=request.query,
                budget_chars=request.budget_chars,
                limit=request.person_limit,
            )
        return ExperienceRecallResult(person=person, general=tuple(general))
