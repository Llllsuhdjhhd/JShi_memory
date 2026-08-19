"""Shared helpers to capture recall traces for scenario scripts.

2026.06: tri-band single stream. Legacy dual-stream hooks removed from default path;
import from ``tests.scenarios.common.recall_hooks`` for new runners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from rems.services.recall_service import RecallService

from tests.scenarios.common.recall_hooks import install_recall_hooks as install_recall_hooks


def recall_block_to_json(block) -> list[dict[str, Any]]:
    return [
        {
            "event_id": it.event_id,
            "summary_level": it.summary_level,
            "score": it.score,
            "content_preview": (it.content or "")[:500],
            "content_chars": len(it.content or ""),
        }
        for it in block.items
    ]


def latest_recall_log_row(recall_log_repo) -> dict[str, Any] | None:
    rows = recall_log_repo.list_all()
    if not rows:
        return None
    rid, eids = rows[-1]
    return {"recall_id": rid, "event_ids": list(eids)}


def write_recall_trace_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
