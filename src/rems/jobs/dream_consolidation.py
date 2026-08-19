from __future__ import annotations

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..skills.dream_evolution import run_dream_consolidation
from ..storage.database import Database
from ..storage.repository import EventRepository


def main() -> None:
    config = REMSConfig()
    db = Database(config.storage.database_url)
    event_repo = EventRepository(db)
    llm = LLMProvider(config)
    run_dream_consolidation(config, llm, event_repo)


if __name__ == "__main__":
    main()
