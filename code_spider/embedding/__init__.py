"""Embedding providers.

Default: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim) running in-process.
A :class:`HashEmbeddingProvider` exists for offline CI / tests. The
``litellm`` provider routes through the LiteLLM SDK to any supported cloud
provider (Voyage, OpenAI, Cohere, …) — see
:mod:`code_spider.embedding.litellm_provider`.
"""

from code_spider.embedding.provider import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    clear_registry,
    get_embedding_provider,
    register_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "clear_registry",
    "get_embedding_provider",
    "register_embedding_provider",
]
