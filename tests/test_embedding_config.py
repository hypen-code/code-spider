"""Env-driven embedding configuration tests.

Asserts that ``_load_embedding_settings`` reads every documented
``CODE_SPIDER_EMBED_*`` knob with the right type and falls back to safe
defaults when missing or empty.
"""

from __future__ import annotations

import pytest

from code_spider.config import _DEFAULT_EMBED_DIM, _load_embedding_settings


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CODE_SPIDER_EMBED_PROVIDER",
        "CODE_SPIDER_EMBED_MODEL",
        "CODE_SPIDER_EMBED_DIM",
        "CODE_SPIDER_EMBED_BATCH_SIZE",
        "CODE_SPIDER_EMBED_API_BASE",
        "CODE_SPIDER_EMBED_API_KEY",
        "CODE_SPIDER_EMBED_TIMEOUT_S",
        "CODE_SPIDER_EMBED_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = _load_embedding_settings()
    assert settings.provider == "sentence-transformers"
    assert settings.model is None
    assert settings.dim == _DEFAULT_EMBED_DIM == 384
    assert settings.batch_size == 64
    assert settings.api_base is None
    assert settings.api_key is None
    assert settings.timeout_s == 30.0
    assert settings.max_retries == 3


def test_litellm_voyage_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("CODE_SPIDER_EMBED_PROVIDER", "litellm")
    monkeypatch.setenv("CODE_SPIDER_EMBED_MODEL", "voyage/voyage-code-3")
    monkeypatch.setenv("CODE_SPIDER_EMBED_DIM", "1024")
    monkeypatch.setenv("CODE_SPIDER_EMBED_BATCH_SIZE", "16")
    monkeypatch.setenv("CODE_SPIDER_EMBED_TIMEOUT_S", "10.5")
    monkeypatch.setenv("CODE_SPIDER_EMBED_MAX_RETRIES", "5")
    settings = _load_embedding_settings()
    assert settings.provider == "litellm"
    assert settings.model == "voyage/voyage-code-3"
    assert settings.dim == 1024
    assert settings.batch_size == 16
    assert settings.timeout_s == 10.5
    assert settings.max_retries == 5


def test_openrouter_api_base_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("CODE_SPIDER_EMBED_PROVIDER", "litellm")
    monkeypatch.setenv("CODE_SPIDER_EMBED_MODEL", "openrouter/openai/text-embedding-3-small")
    monkeypatch.setenv("CODE_SPIDER_EMBED_DIM", "1536")
    monkeypatch.setenv("CODE_SPIDER_EMBED_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("CODE_SPIDER_EMBED_API_KEY", "sk-or-secret")
    settings = _load_embedding_settings()
    assert settings.api_base == "https://openrouter.ai/api/v1"
    assert settings.api_key == "sk-or-secret"
    assert settings.dim == 1536


def test_invalid_int_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("CODE_SPIDER_EMBED_DIM", "not-an-int")
    with pytest.raises(RuntimeError, match="must be an integer"):
        _load_embedding_settings()


def test_invalid_float_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("CODE_SPIDER_EMBED_TIMEOUT_S", "soon")
    with pytest.raises(RuntimeError, match="must be a float"):
        _load_embedding_settings()


def test_empty_provider_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank ``CODE_SPIDER_EMBED_PROVIDER`` must not result in the empty
    string being treated as a provider name — fall back to the default."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("CODE_SPIDER_EMBED_PROVIDER", "")
    settings = _load_embedding_settings()
    assert settings.provider == "sentence-transformers"
