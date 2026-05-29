"""Embedding provider tests — deterministic HashEmbeddingProvider only."""

from __future__ import annotations

import math

from code_spider.embedding import HashEmbeddingProvider, get_embedding_provider


def test_hash_provider_returns_correct_dim() -> None:
    p = HashEmbeddingProvider()
    vecs = p.embed_batch(["hello", "world"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == p.dim == 384


def test_hash_provider_vectors_are_l2_normalised() -> None:
    p = HashEmbeddingProvider()
    [v] = p.embed_batch(["any non-empty text"])
    norm = math.sqrt(sum(x * x for x in v))
    assert 0.99 < norm < 1.01


def test_hash_provider_is_deterministic() -> None:
    p = HashEmbeddingProvider()
    a = p.embed_batch(["abc"])[0]
    b = p.embed_batch(["abc"])[0]
    assert a == b


def test_hash_provider_distinct_texts_have_different_vectors() -> None:
    p = HashEmbeddingProvider()
    [a, b] = p.embed_batch(["hello", "world"])
    assert a != b


def test_provider_registry_returns_hash_provider() -> None:
    provider = get_embedding_provider("hash")
    assert provider.name == "hash"
    assert provider.dim == 384
