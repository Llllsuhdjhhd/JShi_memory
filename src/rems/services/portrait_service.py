"""Read a descriptive portrait produced by the white-painting batch builder."""

from __future__ import annotations

from ..config import REMSConfig
from ..models.portrait import MemoryFatigue, ObjectPortrait
from ..storage.repository import PortraitRepository


class PortraitService:
    """Existing backend read surface; formation is a separate batch operation."""

    def __init__(self, config: REMSConfig, repo: PortraitRepository):
        self._config = config
        self._repo = repo

    def event_fatigue(self) -> MemoryFatigue:
        return MemoryFatigue.manual(self._config.event_fatigue)

    def portrait_fatigue(self, object_burden: float | None = None) -> MemoryFatigue:
        return MemoryFatigue.manual(self._config.portrait_fatigue)

    def get_portrait(self, object_id: str) -> ObjectPortrait | None:
        return self._repo.get_portrait(object_id)


__all__ = ["PortraitService"]
