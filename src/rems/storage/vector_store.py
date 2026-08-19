from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..config import REMSConfig
from ..embedding.tri_band import TriBandEncoder, TriBandVectors
from ..models.event import Event
from ..storage.qdrant_store import QdrantStore
from ..strategies.event_weight import event_id_to_point_id

if TYPE_CHECKING:
    from ..observability import PerfMonitor

logger = logging.getLogger(__name__)


class VectorStore:
    """Facade over Qdrant tri-band store + encoder (§4.4.0)."""

    def __init__(
        self,
        config: REMSConfig,
        perf_monitor: "PerfMonitor | None" = None,
        *,
        qdrant: QdrantStore | None = None,
        tri_band: TriBandEncoder | None = None,
    ):
        self._config = config
        self._perf = perf_monitor
        self._qdrant = qdrant or QdrantStore(config, perf_monitor=perf_monitor)
        self._tri_band = tri_band

    @property
    def qdrant(self) -> QdrantStore:
        return self._qdrant

    def set_tri_band(self, encoder: TriBandEncoder) -> None:
        self._tri_band = encoder

    def upsert_event_vectors(self, event: Event) -> None:
        if self._tri_band is None:
            raise RuntimeError("TriBandEncoder not configured on VectorStore")
        vectors = self._tri_band.encode(event)
        role_ids = [r.role_id for r in event.role_list]
        payload = {
            "event_id": event.event_id,
            "is_abstract": event.is_abstract,
            "status": event.status.value,
            "role_ids": ",".join(role_ids),
        }
        self._qdrant.upsert_event(event_id_to_point_id(event.event_id), vectors, payload)

    def search_tri_band(
        self,
        query: TriBandVectors,
        *,
        weights: tuple[float, float, float] | None = None,
        filter_ids: list[str] | None = None,
        n_results: int = 60,
        bypass_filter: bool = False,
    ) -> list[dict[str, Any]]:
        w = weights or self._config.tri_band.weight_fact
        if not self._config.tri_band.emo_search_enabled:
            w = (w[0], 0.0, w[2])
            total = w[0] + w[2] or 1.0
            w = (w[0] / total, 0.0, w[2] / total)
        point_ids = None
        if filter_ids and not bypass_filter:
            point_ids = [event_id_to_point_id(eid) for eid in filter_ids]
        hits = self._qdrant.search(
            query,
            weights=w,
            filter_ids=point_ids,
            n_results=n_results,
            bypass_filter=bypass_filter,
        )
        return hits

    def add_event(
        self,
        event_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Legacy: build minimal event and upsert act-only via tri-band if encoder present."""
        if self._tri_band is None:
            return
        ev = Event(event_id=event_id, content_raw=text)
        self.upsert_event_vectors(ev)

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._tri_band is None:
            return []
        qv = self._tri_band.encode_query(query)
        hits = self.search_tri_band(qv, n_results=n_results)
        return [{"event_id": h["event_id"], "distance": h.get("distance", 1.0), "metadata": {}} for h in hits]

    def delete_event(self, event_id: str) -> None:
        self._qdrant.delete_event(event_id_to_point_id(event_id))

    def count(self) -> int:
        return self._qdrant.count()

    def cosine_between_events(self, event_id_a: str, event_id_b: str) -> float:
        return self._qdrant.cosine_between_events(
            event_id_to_point_id(event_id_a),
            event_id_to_point_id(event_id_b),
        )

    def add_white_painting(self, role_id: str, event_id: str, text: str, metadata: dict | None = None) -> None:
        """Deprecated: stream B removed; no-op for backward compatibility."""

    def search_white_paintings(self, query: str, n_results: int = 10, where: dict | None = None) -> list[dict]:
        """Deprecated: use tri-band entity weight instead."""
        return []

    def delete_white_painting(self, role_id: str, event_id: str) -> None:
        pass
