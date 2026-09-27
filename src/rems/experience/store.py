"""Dedicated persistence for person interaction experience and its evidence."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from ..models.portrait import PendingPortraitSummary
from ..storage.database import (
    Database,
    PersonExperienceEvidenceRecord,
    PersonExperiencePortraitRecord,
    PersonExperienceReviewRecord,
    PersonExperienceRevisionRecord,
)
from .models import (
    ContextualAdjustment,
    PersonExperiencePortrait,
    PersonExperienceRevision,
    PortraitRevisionKind,
    RevisedClaim,
)


class PersonExperienceRepository:
    """Store interaction portraits separately from descriptive portraits."""

    def __init__(self, db: Database):
        self._db = db

    def get_portrait(self, object_id: str) -> PersonExperiencePortrait | None:
        with self._db.session() as session:
            row = session.get(PersonExperiencePortraitRecord, object_id)
            if row is None:
                return None
            shape = row.shape or {}
            return PersonExperiencePortrait(
                subject_id=row.subject_id,
                object_id=row.object_id,
                content=row.content,
                brief=row.brief,
                baseline=shape.get("baseline") or row.content,
                adjustments=tuple(
                    ContextualAdjustment(**item)
                    for item in shape.get("adjustments", [])
                ),
                uncertainties=tuple(shape.get("uncertainties", [])),
                version=int(row.version or 1),
                source_chars=row.total_content_len,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    def add_summary(self, item: PendingPortraitSummary) -> None:
        with self._db.session() as session:
            session.merge(PersonExperienceEvidenceRecord(
                summary_id=item.summary_id,
                subject_id=item.subject_id,
                object_id=item.object_id,
                text=item.text,
                source_event_id=item.source_event_id,
                weight=item.weight,
                incorporated=item.incorporated,
                created_at=item.created_at,
            ))
            session.commit()

    def list_summaries(
        self, object_id: str, *, incorporated: bool | None = None
    ) -> list[PendingPortraitSummary]:
        with self._db.session() as session:
            query = session.query(PersonExperienceEvidenceRecord).filter(
                PersonExperienceEvidenceRecord.object_id == object_id
            )
            if incorporated is not None:
                query = query.filter(
                    PersonExperienceEvidenceRecord.incorporated == incorporated
                )
            rows = query.order_by(PersonExperienceEvidenceRecord.created_at).all()
            return [PendingPortraitSummary(
                summary_id=row.summary_id,
                subject_id=row.subject_id,
                object_id=row.object_id,
                text=row.text,
                source_event_id=row.source_event_id,
                weight=row.weight,
                incorporated=row.incorporated,
                created_at=row.created_at,
            ) for row in rows]

    def last_insufficient_signature(self, subject_id: str, object_id: str) -> str | None:
        with self._db.session() as session:
            row = session.get(PersonExperienceReviewRecord, object_id)
            if row is None:
                return None
            if row.subject_id != subject_id:
                raise ValueError("experience review belongs to a different subject")
            return row.evidence_signature

    def record_insufficient(
        self, subject_id: str, object_id: str, evidence_signature: str
    ) -> None:
        with self._db.session() as session:
            previous = session.get(PersonExperienceReviewRecord, object_id)
            if previous is not None and previous.subject_id != subject_id:
                raise ValueError("experience review belongs to a different subject")
            session.merge(PersonExperienceReviewRecord(
                subject_id=subject_id,
                object_id=object_id,
                evidence_signature=evidence_signature,
                reviewed_at=datetime.now(),
            ))
            session.commit()

    def save_and_mark(
        self,
        portrait: PersonExperiencePortrait,
        summary_ids: list[str],
        *,
        kind: PortraitRevisionKind,
        reason: str,
        revised_claims: tuple[RevisedClaim, ...] = (),
    ) -> None:
        """Publish a snapshot, revision, and evidence consumption atomically."""
        with self._db.session() as session:
            shape = {
                "baseline": portrait.baseline,
                "adjustments": [
                    {"context": item.context, "approach": item.approach}
                    for item in portrait.adjustments
                ],
                "uncertainties": list(portrait.uncertainties),
            }
            session.merge(PersonExperiencePortraitRecord(
                object_id=portrait.object_id,
                subject_id=portrait.subject_id,
                content=portrait.content,
                brief=portrait.brief,
                shape=shape,
                version=portrait.version,
                total_content_len=portrait.source_chars,
                created_at=portrait.created_at,
                updated_at=portrait.updated_at,
            ))
            event_ids: list[str] = []
            for summary_id in summary_ids:
                row = session.get(PersonExperienceEvidenceRecord, summary_id)
                if (
                    row is None or row.subject_id != portrait.subject_id
                    or row.object_id != portrait.object_id
                ):
                    raise ValueError(f"invalid portrait evidence: {summary_id}")
                row.incorporated = True
                if row.source_event_id and row.source_event_id not in event_ids:
                    event_ids.append(row.source_event_id)
            session.add(PersonExperienceRevisionRecord(
                revision_id=f"XPR-{uuid4().hex}",
                subject_id=portrait.subject_id,
                object_id=portrait.object_id,
                version=portrait.version,
                kind=kind,
                reason=reason,
                revised_claims=[
                    {"previous": item.previous, "updated": item.updated, "reason": item.reason}
                    for item in revised_claims
                ],
                evidence_event_ids=event_ids,
                snapshot={"content": portrait.content, "brief": portrait.brief, **shape},
                created_at=datetime.now(),
            ))
            review = session.get(PersonExperienceReviewRecord, portrait.object_id)
            if review is not None:
                session.delete(review)
            session.commit()

    def list_revisions(self, subject_id: str, object_id: str) -> tuple[PersonExperienceRevision, ...]:
        with self._db.session() as session:
            rows = (
                session.query(PersonExperienceRevisionRecord)
                .filter(
                    PersonExperienceRevisionRecord.subject_id == subject_id,
                    PersonExperienceRevisionRecord.object_id == object_id,
                )
                .order_by(PersonExperienceRevisionRecord.created_at)
                .all()
            )
            return tuple(PersonExperienceRevision(
                revision_id=row.revision_id,
                subject_id=row.subject_id,
                object_id=row.object_id,
                version=row.version,
                kind=row.kind,
                reason=row.reason,
                revised_claims=tuple(
                    RevisedClaim(**item) for item in (row.revised_claims or [])
                ),
                evidence_event_ids=tuple(row.evidence_event_ids or []),
                content=(row.snapshot or {}).get("content", ""),
                brief=(row.snapshot or {}).get("brief", ""),
                baseline=(row.snapshot or {}).get("baseline", ""),
                adjustments=tuple(ContextualAdjustment(**item) for item in (
                    (row.snapshot or {}).get("adjustments", [])
                )),
                uncertainties=tuple((row.snapshot or {}).get("uncertainties", [])),
                created_at=row.created_at,
            ) for row in rows)
