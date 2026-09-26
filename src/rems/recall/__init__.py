"""回忆模块（design/1010）：可插拔 provider + 检索管线。

按 design/1010：事件与对象事实两路材料，语义和词面入候选，可及性只调整接近项的顺序。
"""

from .pipeline import RecallPipeline, material_source_hash, retrieval_text
from .providers import (
    BgeReranker,
    EmbeddingProvider,
    HashEmbedding,
    NullReranker,
    QdrantRecallVectorStore,
    RecallVectorStore,
    RerankerProvider,
    SentenceTransformerEmbedding,
    versioned_collection_name,
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
    "material_source_hash",
    "retrieval_text",
    "versioned_collection_name",
]
