from __future__ import annotations

from typing import Protocol

from ..models.event import Event
from ..storage.repository import EventRepository


class AbstractionEvidencePolicy(Protocol):
    """Select evidence events used by abstract synthesis.

    The mined subset remains the abstract event's ``source_events``. This policy
    controls what factual material is sent to the synthesis skill.
    """

    def collect(self, events: list[Event], event_repo: EventRepository) -> list[Event]:
        """Return basic events to use as evidence for synthesis."""
        ...


class LeafContentRawEvidencePolicy:
    """Default evidence policy: expand every source to leaf basic events.

    This keeps higher-order abstractions grounded in basic-event ``content_raw``
    rather than repeatedly compressing previous abstract text.
    """

    def collect(self, events: list[Event], event_repo: EventRepository) -> list[Event]:
        basic_ids: list[str] = []
        seen: set[str] = set()
        for event in sorted(events, key=lambda e: (e.create_time, e.event_id)):
            ids = (
                event_repo.resolve_basic_event_ids(event.event_id)
                if event.is_abstract
                else [event.event_id]
            )
            for event_id in ids:
                if event_id in seen:
                    continue
                seen.add(event_id)
                basic_ids.append(event_id)

        basics: list[Event] = []
        for event_id in basic_ids:
            event = event_repo.get(event_id)
            if event is None or event.is_abstract or event.is_tombstoned:
                continue
            basics.append(event)
        basics.sort(key=lambda e: (e.create_time, e.event_id))
        return basics
