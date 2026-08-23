"""回忆模块（design/1010）：可插拔 provider + 检索管线。

旧回忆模块（RecallService / tri_band / retrieval backends / BM25 / Tier-1）不再保留，
完全按 1010 新方案实现：BGE-M3（或本地小模型）嵌入 + 多路召回 + RRF + 可选精排。
"""

from .pipeline import RecallPipeline
from .providers import (
    BgeReranker,
    EmbeddingProvider,
    HashEmbedding,
    NullReranker,
    QdrantRecallVectorStore,
    RecallVectorStore,
    RerankerProvider,
    SentenceTransformerEmbedding,
)

__all__ = [
    "BgeReranker",
    "EmbeddingProvider",
    "HashEmbedding",
    "NullReranker",
    "QdrantRecallVectorStore",
    "RecallPipeline",
    "RecallVectorStore",
    "RerankerProvider",
    "SentenceTransformerEmbedding",
]
