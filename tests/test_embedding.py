"""Embedding provider tests — deterministic HashEmbeddingProvider only."""

from __future__ import annotations

import math

import pytest

from code_spider.config import EmbeddingSettings
from code_spider.embedding import (
    HashEmbeddingProvider,
    clear_registry,
    get_embedding_provider,
)


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


# --------------------------------------------------------------------------- #
# Registry can resolve the litellm provider lazily                            #
# --------------------------------------------------------------------------- #


def test_registry_lazy_loads_litellm_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_embedding_provider('litellm', settings=...)`` does not require
    LiteLLM to be installed, because :class:`LiteLLMEmbeddingProvider` lazily
    imports the SDK on first ``embed_batch``."""
    # Ensure a clean registry; other tests may have registered litellm.
    clear_registry()
    settings = EmbeddingSettings(
        provider="litellm",
        model="voyage/voyage-code-3",
        dim=1024,
        batch_size=64,
        api_base=None,
        api_key=None,
        timeout_s=30.0,
        max_retries=3,
        max_input_chars=120_000,
        workers=1,
    )
    provider = get_embedding_provider("litellm", settings=settings)
    assert provider.name == "litellm"
    assert provider.dim == 1024
    # Clean up so the cached instance doesn't leak into other tests.
    clear_registry()


def test_unknown_provider_raises_key_error() -> None:
    clear_registry()
    with pytest.raises(KeyError):
        get_embedding_provider("does-not-exist")
