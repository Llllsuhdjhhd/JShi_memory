from __future__ import annotations

import random
from datetime import datetime
from typing import Iterable

from ..config import REMSConfig
from .database import Database, EventTier1Record


def compute_sample_key(w_eff: float, sample_u: float | None = None) -> tuple[float, float]:
    """A-Res: S_i = U_i ** (1 / w_eff). Returns (sample_u, sample_key)."""
    u = sample_u if sample_u is not None else random.random()
    u = max(u, 1e-12)
    w = max(float(w_eff), 1e-6)
    return u, u ** (1.0 / w)


class Tier1Store:
    """SQLite-backed Tier-1 metadata for active pool pre-filtering (§4.5)."""

    def __init__(self, db: Database, config: REMSConfig):
        self._db = db
        self._config = config

    def upsert_tier1(
        self,
        event_id: str,
        *,
        w_i: float,
        asf_i: float = 0.0,
        sample_u: float | None = None,
    ) -> None:
        asf_clamped = min(max(float(asf_i), 0.0), 0.999999)
        w_eff = max(float(w_i) * (1.0 - asf_clamped), 1e-6)
        existing_u = self.get_sample_u(event_id)
        u, s_key = compute_sample_key(w_eff, sample_u if sample_u is not None else existing_u)
        now = datetime.now()
        with self._db.session() as session:
            row = session.get(EventTier1Record, event_id)
            if row is None:
                row = EventTier1Record(
                    event_id=event_id,
                    sample_key=s_key,
                    sample_u=u,
                    asf_i=asf_clamped,
                    w_i_cached=float(w_i),
                    w_eff_cached=w_eff,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.sample_key = s_key
                row.sample_u = u
                row.asf_i = asf_clamped
                row.w_i_cached = float(w_i)
                row.w_eff_cached = w_eff
                row.updated_at = now
            session.commit()

    def get_asf(self, event_id: str) -> float:
        with self._db.session() as session:
            row = session.get(EventTier1Record, event_id)
            return float(row.asf_i) if row else 0.0

    def get_sample_u(self, event_id: str) -> float | None:
        with self._db.session() as session:
            row = session.get(EventTier1Record, event_id)
            return float(row.sample_u) if row else None

    def batch_update_asf(self, updates: dict[str, float]) -> None:
        for event_id, asf in updates.items():
            with self._db.session() as session:
                row = session.get(EventTier1Record, event_id)
                if row is None:
                    continue
                w_i = float(row.w_i_cached)
                session.commit()
            self.upsert_tier1(event_id, w_i=w_i, asf_i=asf, sample_u=self.get_sample_u(event_id))

    def fetch_active_ids(self, limit: int) -> list[str]:
        limit = max(1, int(limit))
        with self._db.session() as session:
            rows = (
                session.query(EventTier1Record)
                .order_by(EventTier1Record.sample_key.desc())
                .limit(limit)
                .all()
            )
            return [r.event_id for r in rows]

    def list_all_event_ids(self) -> list[str]:
        with self._db.session() as session:
            rows = session.query(EventTier1Record.event_id).all()
            return [r[0] for r in rows]

    def delete(self, event_id: str) -> None:
        with self._db.session() as session:
            row = session.get(EventTier1Record, event_id)
            if row:
                session.delete(row)
                session.commit()

    def bulk_upsert_from_weights(self, weights: Iterable[tuple[str, float, float]]) -> None:
        for event_id, w_i, asf_i in weights:
            self.upsert_tier1(event_id, w_i=w_i, asf_i=asf_i)
