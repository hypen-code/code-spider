"""Embedding provider protocol + a deterministic provider for tests.

The default production provider is :class:`SentenceTransformerProvider`
(see :mod:`code_spider.embedding.st_provider`); it loads on demand because
``sentence-transformers`` is an optional install.

Adding a new provider (Cohere / OpenAI / etc.): implement
:class:`EmbeddingProvider` and register it via
:func:`register_embedding_provider`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Batched embedding interface returning normalised L2 vectors."""

    name: str
    dim: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


_REGISTRY: dict[str, EmbeddingProvider] = {}


def register_embedding_provider(provider: EmbeddingProvider) -> None:
    _REGISTRY[provider.name] = provider


def get_embedding_provider(name: str = "sentence-transformers") -> EmbeddingProvider:
    if name not in _REGISTRY:
        _lazy_load(name)
    if name not in _REGISTRY:
        raise KeyError(f"no embedding provider registered for '{name}'")
    return _REGISTRY[name]


def _lazy_load(name: str) -> None:
    if name == "sentence-transformers":
        from code_spider.embedding.st_provider import SentenceTransformerProvider

        register_embedding_provider(SentenceTransformerProvider())
    elif name == "hash":
        register_embedding_provider(HashEmbeddingProvider())


class HashEmbeddingProvider:
    """Deterministic provider used in tests and offline CI environments.

    Produces stable, low-quality vectors purely from text hashes — never use
    for real semantic search. Dimension matches ``all-MiniLM-L6-v2``.
    """

    name: str = "hash"
    dim: int = 384

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec: list[float] = [0.0] * self.dim
        if not text:
            vec[0] = 1.0
            return vec
        # blake2b is capped at 64-byte digests; chain seeded hashes until we
        # fill ``self.dim`` bytes. Each seed differs so chunks aren't repeats.
        payload = text.encode("utf-8")
        bytes_needed = self.dim
        buf = bytearray()
        seed = 0
        while len(buf) < bytes_needed:
            chunk = hashlib.blake2b(
                payload,
                digest_size=min(64, bytes_needed - len(buf)),
                person=str(seed).encode("utf-8").ljust(16, b"\x00")[:16],
            ).digest()
            buf.extend(chunk)
            seed += 1
        # Map bytes (0..255) to floats centred around zero.
        for i in range(self.dim):
            vec[i] = (buf[i] - 127.5) / 127.5
        # L2-normalise so cosine similarity behaves predictably.
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
