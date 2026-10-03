"""Build person experience from existing white paintings and read portraits."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from typing import Callable

from ..config import REMSConfig
from ..models.portrait import PendingPortraitSummary
from ..storage.repository import RoleRepository
from .models import (
    ExperiencePortraitSyncResult,
    PersonExperienceFragment,
    PersonExperiencePortrait,
)
from .person_composer import InsufficientEvidence, PersonExperienceComposerPort
from .store import PersonExperienceRepository


class WhitePaintingExperiencePortraitBuilder:
    """An explicit, repeatable white-painting-to-portrait entry point.

    It runs separately from Event ingest and recall. White paintings remain
    evidence; only a composer can turn them into an interaction experience.
    """

    def __init__(
        self,
        config: REMSConfig,
        white_paintings: RoleRepository,
        portraits: PersonExperienceRepository,
        composer: PersonExperienceComposerPort | None,
    ) -> None:
        self._config = config
        self._white_paintings = white_paintings
        self._portraits = portraits
        self._composer = composer

    def sync(self, subject_id: str, object_id: str) -> ExperiencePortraitSyncResult:
        if not subject_id.strip() or not object_id.strip():
            raise ValueError("subject_id and object_id are required")
        portrait = self._portraits.get_portrait(object_id)
        if portrait is not None and portrait.subject_id != subject_id:
            raise ValueError("portrait belongs to a different subject")
        summaries = self._portraits.list_summaries(object_id)
        if any(item.subject_id != subject_id for item in summaries):
            raise ValueError("portrait summaries belong to a different subject")

        known_events = {
            item.source_event_id
            for item in summaries
            if item.summary_id.startswith("xp:") and item.source_event_id
        }
        added = 0
        for painting in self._white_paintings.get_white_painting(object_id):
            if painting.subject_id != subject_id or not painting.role_summary.strip():
                continue
            if not painting.event_id:
                continue
            if painting.event_id in known_events:
                continue
            summary = PendingPortraitSummary(
                summary_id=f"xp:{subject_id}:{object_id}:{painting.event_id}",
                subject_id=subject_id,
                object_id=object_id,
                text=painting.role_summary,
                source_event_id=painting.event_id,
                weight=max(0.0, min(1.0, painting.memory_weight)),
                created_at=painting.create_time,
            )
            self._portraits.add_summary(summary)
            known_events.add(painting.event_id)
            added += 1

        all_summaries = self._portraits.list_summaries(object_id)
        pending = tuple(
            item for item in all_summaries
            if item.subject_id == subject_id
            and item.summary_id.startswith("xp:")
            and not item.incorporated
        )
        pending_chars = sum(len(item.text) for item in pending)
        status = ExperiencePortraitSyncResult(
            object_id=object_id,
            imported_white_paintings=added,
            updated=False,
            pending_white_paintings=len(pending),
            pending_chars=pending_chars,
        )
        count_ready = (
            self._config.experience_person_trigger_count > 0
            and len(pending) >= self._config.experience_person_trigger_count
        )
        chars_ready = (
            self._config.experience_person_trigger_chars > 0
            and pending_chars >= self._config.experience_person_trigger_chars
        )
        if self._composer is None or not (count_ready or chars_ready):
            return status

        # 签名覆盖全部待吸收白描：同一批证据判过一次（不足或无效）就不再调用模型，
        # 有新白描进来再重试。
        signature_source = "|".join(
            [str(portrait.version if portrait else 0), *(item.summary_id for item in pending)]
        )
        signature = sha256(signature_source.encode("utf-8")).hexdigest()
        if self._portraits.last_insufficient_signature(subject_id, object_id) == signature:
            return replace(status, review_outcome="already_reviewed")

        changed_any = False
        outcome = ""
        remaining = pending
        while remaining:
            batch = self._next_batch(remaining)
            outcome, changed = self._absorb(subject_id, object_id, portrait, batch)
            if outcome in {"insufficient", "invalid"}:
                self._portraits.record_insufficient(subject_id, object_id, signature)
                left = remaining
                return replace(
                    status,
                    updated=changed_any,
                    pending_white_paintings=len(left),
                    pending_chars=sum(len(item.text) for item in left),
                    review_outcome=outcome,
                )
            changed_any = changed_any or changed
            portrait = self._portraits.get_portrait(object_id)
            remaining = remaining[len(batch):]
        return ExperiencePortraitSyncResult(object_id, added, changed_any, 0, 0, outcome)

    def _next_batch(
        self, pending: tuple[PendingPortraitSummary, ...]
    ) -> tuple[PendingPortraitSummary, ...]:
        limit = self._config.experience_person_batch_chars
        if limit <= 0:
            return pending
        batch: list[PendingPortraitSummary] = []
        used = 0
        for item in pending:
            if batch and used + len(item.text) > limit:
                break
            batch.append(item)
            used += len(item.text)
        return tuple(batch)

    def _absorb(
        self,
        subject_id: str,
        object_id: str,
        portrait: PersonExperiencePortrait | None,
        pending: tuple[PendingPortraitSummary, ...],
    ) -> tuple[str, bool]:
        """Merge one batch of white paintings into the current portrait."""
        draft = self._composer.compose(
            object_id=object_id,
            previous=portrait,
            pending=pending,
            max_chars=self._config.experience_person_max_chars,
            brief_chars=self._config.experience_person_brief_chars,
        )
        if isinstance(draft, InsufficientEvidence):
            return "insufficient", False
        if draft is None:
            return "invalid", False
        full = draft.full.strip()
        brief = draft.brief.strip()
        if not full or not brief:
            return "invalid", False
        # 修订类别按新旧内容的实际差异定，模型给的标签只作参考：
        # 标签与内容不符时改标签，不丢弃整份可用的草稿。
        draft = replace(draft, revision_kind=_actual_revision_kind(draft, portrait, brief))
        if draft.revision_kind == "confirm" and portrait is not None:
            full = portrait.content
            brief = portrait.brief

        source_chars = (portrait.source_chars if portrait else 0) + sum(
            len(item.text) for item in pending
        )
        now = datetime.now()
        changed = draft.revision_kind != "confirm"
        self._portraits.save_and_mark(PersonExperiencePortrait(
            subject_id=subject_id,
            object_id=object_id,
            content=full,
            brief=brief,
            baseline=draft.baseline,
            adjustments=draft.adjustments,
            uncertainties=draft.uncertainties,
            version=(portrait.version + 1 if portrait and changed else portrait.version if portrait else 1),
            source_chars=source_chars,
            created_at=portrait.created_at if portrait else now,
            updated_at=now if changed else portrait.updated_at,
        ), [item.summary_id for item in pending],
            kind=draft.revision_kind,
            reason=draft.revision_reason,
            revised_claims=draft.revised_claims,
        )
        return draft.revision_kind, changed


def _actual_revision_kind(draft, portrait: PersonExperiencePortrait | None, brief: str) -> str:
    """按新旧内容定修订类别：无旧版为 initial，全同为 confirm，只增为 extend，其余为 revise。"""
    if portrait is None:
        return "initial"
    if (
        draft.baseline == portrait.baseline
        and draft.adjustments == portrait.adjustments
        and draft.uncertainties == portrait.uncertainties
        and brief == portrait.brief
    ):
        return "confirm"
    if draft.revision_kind == "uncertain" and draft.uncertainties:
        return "uncertain"
    if (
        draft.baseline == portrait.baseline
        and set(portrait.adjustments) <= set(draft.adjustments)
        and set(portrait.uncertainties) <= set(draft.uncertainties)
    ):
        return "extend"
    return "revise"


class ExperiencePortraitReader:
    """Resolve an explicitly relevant person's portrait within a char budget."""

    def __init__(
        self,
        portraits: PersonExperienceRepository,
        relevance: Callable[[str, str, str], float] | None = None,
    ):
        self._portraits = portraits
        self._relevance = relevance

    def recall(
        self,
        subject_id: str,
        object_ids: tuple[str, ...],
        *,
        query: str = "",
        budget_chars: int,
        limit: int,
    ) -> tuple[PersonExperienceFragment, ...]:
        if budget_chars <= 0 or limit <= 0:
            return ()
        remaining = budget_chars
        out: list[PersonExperienceFragment] = []
        for object_id in dict.fromkeys(object_ids):
            if len(out) >= limit:
                break
            portrait = self._portraits.get_portrait(object_id)
            if portrait is None or portrait.subject_id != subject_id:
                continue
            score = (
                self._relevance(query, object_id, portrait.content)
                if self._relevance is not None
                else 1.0
            )
            if score <= 0:
                continue
            chosen = next(((name, content) for name, content in (
                ("full", portrait.content), ("brief", portrait.brief),
            ) if content.strip() and len(content) <= remaining), None)
            if chosen is None:
                continue
            evidence = tuple(dict.fromkeys(
                item.source_event_id
                for item in self._portraits.list_summaries(object_id, incorporated=True)
                if (
                    item.subject_id == subject_id
                    and item.summary_id.startswith("xp:")
                    and item.source_event_id
                )
            ))
            variant, text = chosen
            out.append(PersonExperienceFragment(
                object_id=object_id,
                content=text,
                variant=variant,
                source_event_ids=evidence,
                updated_at=portrait.updated_at,
                score=score,
            ))
            remaining -= len(text)
        return tuple(out)
