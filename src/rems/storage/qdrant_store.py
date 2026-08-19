from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import REMSConfig
from ..embedding.tri_band import TriBandVectors

if TYPE_CHECKING:
    from ..observability import PerfMonitor

from ..strategies.event_weight import event_id_to_point_id

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant-backed tri-band vector index (§4.4.0 方案 A)."""

    VECTOR_NAMES = ("vector_act", "vector_emo", "vector_ent")

    def __init__(self, config: REMSConfig, perf_monitor: "PerfMonitor | None" = None):
        self._config = config
        self._perf = perf_monitor
        tb = config.tri_band
        self._dims = {
            "vector_act": tb.vector_dim_act,
            "vector_emo": tb.vector_dim_emo,
            "vector_ent": tb.vector_dim_ent,
        }
        storage = config.storage
        if storage.qdrant_path:
            self._client = QdrantClient(path=storage.qdrant_path)
        elif storage.qdrant_url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=storage.qdrant_url)
        self._collection = storage.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collections = self._client.get_collections().collections or []
            existing = {c.name for c in collections}
        except Exception:
            existing = set()
        if self._collection in existing:
            return
        vectors_config = {
            name: qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            for name, dim in self._dims.items()
        }
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=vectors_config,
        )

    def upsert_event(
        self,
        point_id: str,
        vectors: TriBandVectors,
        payload: dict[str, Any] | None = None,
    ) -> None:
        pl = dict(payload or {})
        pl.setdefault("point_id", point_id)
        point = qmodels.PointStruct(
            id=point_id,
            vector={
                "vector_act": vectors.vector_act,
                "vector_emo": vectors.vector_emo,
                "vector_ent": vectors.vector_ent,
            },
            payload=pl,
        )
        self._client.upsert(collection_name=self._collection, points=[point])

    def search(
        self,
        query: TriBandVectors,
        *,
        weights: tuple[float, float, float] = (0.7, 0.1, 0.2),
        filter_ids: list[str] | None = None,
        n_results: int = 60,
        bypass_filter: bool = False,
    ) -> list[dict[str, Any]]:
        if self._perf is not None:
            with self._perf.timer("rag_search"):
                return self._search_inner(query, weights, filter_ids, n_results, bypass_filter)
        return self._search_inner(query, weights, filter_ids, n_results, bypass_filter)

    def _search_inner(
        self,
        query: TriBandVectors,
        weights: tuple[float, float, float],
        filter_ids: list[str] | None,
        n_results: int,
        bypass_filter: bool,
    ) -> list[dict[str, Any]]:
        alpha, beta, gamma = weights
        total_w = alpha + beta + gamma or 1.0
        alpha, beta, gamma = alpha / total_w, beta / total_w, gamma / total_w

        q_filter = None
        if filter_ids and not bypass_filter:
            q_filter = qmodels.Filter(
                must=[
                    qmodels.HasIdCondition(has_id=[str(i) for i in filter_ids]),
                ]
            )

        # Prefetch each band then fuse with RRF-style merge in application layer.
        band_hits: dict[str, list[tuple[str, float]]] = {}
        band_specs: list[tuple[str, float, list[float]]] = [
            ("vector_act", alpha, query.vector_act),
        ]
        if beta > 0:
            band_specs.append(("vector_emo", beta, query.vector_emo))
        band_specs.append(("vector_ent", gamma, query.vector_ent))
        for name, weight, qvec in band_specs:
            if weight <= 0:
                continue
            response = self._client.query_points(
                collection_name=self._collection,
                query=qvec,
                using=name,
                query_filter=q_filter,
                limit=n_results,
                with_payload=True,
            )
            results = response.points or []
            band_hits[name] = [(str(r.id), float(r.score), r.payload or {}) for r in results]

        merged: dict[str, tuple[float, dict]] = {}
        k = 60
        for hits in band_hits.values():
            for rank, (pid, _score, payload) in enumerate(hits, start=1):
                eid = (payload or {}).get("event_id") or pid
                prev = merged.get(eid, (0.0, payload))
                merged[eid] = (prev[0] + 1.0 / (k + rank), payload)

        sorted_ids = sorted(merged.items(), key=lambda x: x[1][0], reverse=True)[:n_results]
        out: list[dict[str, Any]] = []
        for eid, (score, _payload) in sorted_ids:
            out.append({
                "event_id": eid,
                "score": score,
                "distance": max(0.0, 1.0 - score),
            })
        return out

    def get_vector_act(self, event_id: str) -> list[float] | None:
        pid = event_id_to_point_id(event_id)
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[pid],
            with_vectors=True,
        )
        if not points:
            return None
        vec = points[0].vector
        if isinstance(vec, dict):
            return list(vec.get("vector_act") or [])
        return None

    def cosine_between_events(self, event_id_a: str, event_id_b: str) -> float:
        va = self.get_vector_act(event_id_a)
        vb = self.get_vector_act(event_id_b)
        if not va or not vb:
            return 0.0
        denom = (math.sqrt(sum(x * x for x in va)) * math.sqrt(sum(y * y for y in vb))) or 1.0
        return sum(x * y for x, y in zip(va, vb)) / denom

    def delete_event(self, event_id: str) -> None:
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.PointIdsList(points=[event_id]),
            )
        except Exception:
            logger.warning("Failed to delete %s from Qdrant", event_id)

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return int(info.points_count or 0)
