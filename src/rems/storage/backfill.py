"""Backfill helpers for legacy memory data (design/1010)."""

from __future__ import annotations

from rems.services.event_service import initial_forgetting_factor
from rems.storage.database import Database, EventRecord


def backfill_event_forgetting_factors(
    database: Database,
    gain: float = 2.0,
    *,
    dry_run: bool = False,
) -> int:
    """Initialize forgetting_factor for legacy events.

    Only touches rows with activation_energy > 0 and forgetting_factor == 1.0,
    so reinforced (already-changed) values are never overwritten. Idempotent.
    """
    updated = 0
    with database.session() as session:
        rows = (
            session.query(EventRecord)
            .filter(
                EventRecord.activation_energy > 0,
                EventRecord.forgetting_factor == 1.0,
            )
            .all()
        )
        for rec in rows:
            new_factor = initial_forgetting_factor(rec.activation_energy, gain)
            if dry_run:
                print(
                    f"[dry-run] {rec.event_id}: activation={rec.activation_energy:.3f} "
                    f"factor 1.0 -> {new_factor:.3f}"
                )
            else:
                rec.forgetting_factor = new_factor
            updated += 1
        if not dry_run:
            session.commit()
    return updated


def reset_event_forgetting_factors(
    database: Database,
    gain: float = 2.0,
    *,
    dry_run: bool = False,
) -> int:
    """Reset forgetting_factor to the activation-derived initial value.

    清除 recall 强化（_reinforce ×1.5 累积）造成的污染，恢复初始强度；
    幂等，且不触碰 activation_energy <= 0 的行（保持默认 1.0）。
    """
    updated = 0
    with database.session() as session:
        rows = session.query(EventRecord).filter(
            EventRecord.activation_energy > 0
        ).all()
        for rec in rows:
            init = initial_forgetting_factor(rec.activation_energy, gain)
            if abs((rec.forgetting_factor or 1.0) - init) < 1e-9:
                continue
            if dry_run:
                print(
                    f"[dry-run] {rec.event_id}: factor {rec.forgetting_factor:.4f} "
                    f"-> {init:.4f}"
                )
            else:
                rec.forgetting_factor = init
            updated += 1
        if not dry_run:
            session.commit()
    return updated
