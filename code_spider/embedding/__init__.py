"""Embedding providers.

Default: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim) running in-process.
A :class:`HashEmbeddingProvider` exists for offline CI / tests.
"""

from code_spider.embedding.provider import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    get_embedding_provider,
    register_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "get_embedding_provider",
    "register_embedding_provider",
]
