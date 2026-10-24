"""回忆可插拔 provider（docs/project-overview.md）：Embedding / Reranker / VectorStore 接口与本地实现。"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """文本嵌入接口（本地模型 / API / hash 均可插拔）。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    """精排接口（cross-encoder / 无精排时用 NullReranker）。"""

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]: ...


def versioned_collection_name(base: str, version: int) -> str:
    """嵌入版本变了就换集合名，旧集合留在原地。"""
    suffix = f"_emb{int(version)}"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


class RecallVectorStore(Protocol):
    """回忆向量库接口（Qdrant 本地 / 远程 / 内存均可插拔）。"""

    def upsert(self, event_id: str, vector: list[float], payload: dict) -> None: ...

    def get(self, logical_id: str) -> dict | None: ...

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        payload_filter: dict | None = None,
        any_of: dict[str, list] | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict]: ...

    def delete(self, event_id: str) -> None: ...

    def list_event_logical_ids(self, event_id: str) -> list[str]: ...


class HashEmbedding:
    """确定性 hash 嵌入：离线测试 / 无模型环境用。"""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def _vec(self, text: str) -> list[float]:
        out = [0.0] * self._dim
        for i, ch in enumerate(text):
            h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
            out[h % self._dim] += 1.0
        norm = sum(v * v for v in out) ** 0.5 or 1.0
        return [v / norm for v in out]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class SentenceTransformerEmbedding:
    """本地 sentence-transformers 嵌入（bge-base-zh / BGE-M3 等）。"""

    def __init__(self, model_name: str, query_instruction: str = ""):
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            self._ensure()
            return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    def split_document(self, text: str) -> list[str]:
        """按实际编码器 token 上限分块；保留原文切片和相邻 32 token 重叠。"""
        with self._lock:
            self._ensure()
            tokenizer = self._model.tokenizer
            capacity = int(self._model.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False)
            if capacity < 2:
                raise ValueError("Encoder document capacity is too small")
            offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)["offset_mapping"]
            if not offsets:
                return [text]
            chunks = []
            start = 0
            while start < len(offsets):
                end = min(start + capacity, len(offsets))
                left = 0 if start == 0 else offsets[start][0]
                right = len(text) if end == len(offsets) else offsets[end][0]
                chunk = text[left:right]
                # Re-tokenizing a substring can change a boundary wordpiece.
                while len(tokenizer(chunk, add_special_tokens=True)["input_ids"]) > self._model.max_seq_length:
                    end -= 1
                    if end <= start:
                        raise ValueError("Unable to form an encoder-sized raw chunk")
                    right = offsets[end][0]
                    chunk = text[left:right]
                chunks.append(chunk)
                if end == len(offsets):
                    break
                start = max(start + 1, end - min(32, capacity // 4))
            return chunks

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            self._ensure()
            return self._model.encode(
                [self._query_instruction + text], normalize_embeddings=True
            )[0].tolist()


class NullReranker:
    """无精排：保持候选顺序（纯 RRF 场景）。"""

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        return [1.0] * len(candidates)


class BgeReranker:
    """本地 bge-reranker cross-encoder 精排。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        with self._lock:
            self._ensure()
            pairs = [(query, c) for c in candidates]
            scores = self._model.predict(pairs)
            return [float(s) for s in scores]


class QdrantRecallVectorStore:
    """Qdrant 本地 / 内存向量库（qdrant-client 本地模式，无需 docker）。"""

    def __init__(
        self,
        collection: str = "events",
        *,
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
    ):
        from qdrant_client import QdrantClient

        if path:
            self._client = QdrantClient(path=path)
        elif url and url != ":memory:":
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(":memory:")
        self._collection = collection
        self._dim: int | None = None
        # 同实例并发 upsert/search：本地 Qdrant 无细粒度内部锁，窄锁保护。
        self._lock = threading.RLock()

    def _ensure_collection(self, dim: int) -> None:
        if self._dim == dim:
            return
        from qdrant_client import models

        exists = False
        try:
            exists = self._client.collection_exists(self._collection)
        except Exception:  # noqa: BLE001
            exists = False
        if exists:
            info = self._client.get_collection(self._collection)
            size = int(info.config.params.vectors.size)
            if size != dim:
                raise RuntimeError(
                    f"集合 {self._collection} 的向量维度是 {size}，当前嵌入维度是 {dim}。"
                    "请提高 recall_embedding_version 后使用新集合，旧集合保留。"
                )
            self._dim = dim
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self._dim = dim

    def upsert(self, event_id: str, vector: list[float], payload: dict) -> None:
        with self._lock:
            from qdrant_client import models

            self._ensure_collection(len(vector))
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=hashlib.md5(event_id.encode("utf-8")).hexdigest(),
                        vector=vector,
                        payload={"event_id": event_id, **payload},
                    )
                ],
            )

    def get(self, logical_id: str) -> dict | None:
        with self._lock:
            try:
                if not self._client.collection_exists(self._collection):
                    return None
            except Exception:  # noqa: BLE001
                return None
            found = self._client.retrieve(
                collection_name=self._collection,
                ids=[self._point_id(logical_id)],
                with_payload=True,
                with_vectors=False,
            )
            if not found or not found[0].payload:
                return None
            return dict(found[0].payload)

    def list_event_logical_ids(self, event_id: str) -> list[str]:
        """Return event and fact point keys currently stored for one event."""
        with self._lock:
            from qdrant_client import models

            if not self._client.collection_exists(self._collection):
                return []
            condition = models.Filter(must=[models.FieldCondition(
                key="event_id", match=models.MatchValue(value=event_id),
            )])
            found: list[str] = []
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=condition,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    if payload.get("type") == "event":
                        index = int(payload.get("chunk_index") or 0)
                        found.append(event_id if index == 0 else f"{event_id}#raw#{index}")
                    elif payload.get("type") == "object_event_fact" and payload.get("object_id"):
                        found.append(f"{event_id}#fact#{payload['object_id']}")
                    elif payload.get("type") == "event_detail" and payload.get("source_event_id"):
                        found.append(f"{event_id}#detail#{payload['source_event_id']}")
                if offset is None:
                    break
            return found

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        payload_filter: dict | None = None,
        any_of: dict[str, list] | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict]:
        with self._lock:
            from qdrant_client import models

            self._ensure_collection(len(vector))
            conditions = []
            for key, value in (payload_filter or {}).items():
                conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                ))
            for key, values in (any_of or {}).items():
                if not values:
                    continue
                conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=list(values)),
                ))
            for key, bounds in (ranges or {}).items():
                start, end = bounds
                conditions.append(models.FieldCondition(
                    key=key,
                    range=models.Range(gte=start, lte=end),
                ))
            qfilter = models.Filter(must=conditions) if conditions else None
            hits: list[dict] = []
            if hasattr(self._client, "query_points"):
                resp = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    query_filter=qfilter,
                    limit=top_k,
                    with_payload=True,
                )
                for h in resp.points:
                    if h.payload:
                        hits.append({
                            "event_id": h.payload.get("event_id"),
                            "score": float(h.score),
                            "payload": h.payload,
                        })
            else:
                raw = self._client.search(
                    collection_name=self._collection,
                    query_vector=vector,
                    query_filter=qfilter,
                    limit=top_k,
                    with_payload=True,
                )
                hits = [
                    {"event_id": h.payload.get("event_id"), "score": float(h.score), "payload": h.payload}
                    for h in raw
                    if h.payload
                ]
            return hits

    def delete(self, event_id: str) -> None:
        with self._lock:
            try:
                if not self._client.collection_exists(self._collection):
                    return
            except Exception:  # noqa: BLE001
                return
            self._client.delete(
                collection_name=self._collection,
                points_selector=[self._point_id(event_id)],
            )

    @staticmethod
    def _point_id(logical_id: str) -> str:
        return hashlib.md5(logical_id.encode("utf-8")).hexdigest()
