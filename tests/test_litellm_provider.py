"""Tests for :class:`LiteLLMEmbeddingProvider` with a stubbed SDK.

We don't actually call any cloud provider — the LiteLLM ``embedding`` callable
is injected at construction so every code path runs against deterministic
fixtures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from code_spider.config import EmbeddingSettings
from code_spider.embedding.litellm_provider import (
    LiteLLMEmbeddingProvider,
    _is_payload_too_large,
    _normalise,
)


def _settings(
    *,
    model: str | None = "voyage/voyage-code-3",
    dim: int = 4,
    batch_size: int = 2,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 30.0,
    max_retries: int = 3,
    max_input_chars: int = 120_000,
) -> EmbeddingSettings:
    return EmbeddingSettings(
        provider="litellm",
        model=model,
        dim=dim,
        batch_size=batch_size,
        api_base=api_base,
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_input_chars=max_input_chars,
    )


@dataclass
class _StubResponse:
    """Mimics the dict-like shape LiteLLM exposes for embedding responses."""

    data: list[dict[str, object]]

    def __getitem__(self, key: str) -> object:
        return self.data if key == "data" else None  # pragma: no cover


class _StubEmbed:
    """Configurable stand-in for ``litellm.embedding``."""

    def __init__(self, *, dim: int) -> None:
        self.calls: list[dict[str, object]] = []
        self._dim = dim

    def __call__(self, **kwargs: object) -> _StubResponse:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        # Produce deterministic per-input vectors so the test can assert
        # ordering, batching and normalisation.
        data: list[dict[str, object]] = []
        for offset, text in enumerate(inputs):
            base = float(len(text) + offset)
            vec = [base, base + 1.0, base + 2.0, base + 3.0][: self._dim]
            # Pad / clip to ``self._dim`` so the test can also exercise the
            # dim-mismatch path by passing a different model.
            while len(vec) < self._dim:
                vec.append(1.0)
            data.append({"embedding": vec, "index": offset})
        return _StubResponse(data=data)


# --------------------------------------------------------------------------- #
# Constructor + plumbing                                                      #
# --------------------------------------------------------------------------- #


class TestConstructor:
    def test_requires_model(self) -> None:
        with pytest.raises(ValueError, match="CODE_SPIDER_EMBED_MODEL"):
            LiteLLMEmbeddingProvider(_settings(model=None))

    def test_name_and_dim_match_settings(self) -> None:
        provider = LiteLLMEmbeddingProvider(_settings(dim=1024), embedding_fn=_StubEmbed(dim=1024))
        assert provider.name == "litellm"
        assert provider.dim == 1024


# --------------------------------------------------------------------------- #
# Embed batch behaviour                                                       #
# --------------------------------------------------------------------------- #


class TestEmbedBatch:
    def test_empty_input_skips_sdk_call(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=4), embedding_fn=stub)
        assert provider.embed_batch([]) == []
        assert stub.calls == []

    def test_batches_respect_batch_size(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=4, batch_size=2), embedding_fn=stub)
        out = provider.embed_batch(["a", "bb", "ccc", "dddd", "eeeee"])
        assert len(out) == 5
        # Five inputs, batch_size=2 → ceil(5/2) = 3 SDK calls.
        assert len(stub.calls) == 3
        sizes = [len(call["input"]) for call in stub.calls]  # type: ignore[arg-type]
        assert sizes == [2, 2, 1]

    def test_vectors_are_l2_normalised(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=4), embedding_fn=stub)
        [v] = provider.embed_batch(["hello"])
        norm = math.sqrt(sum(x * x for x in v))
        assert 0.999 < norm < 1.001

    def test_passes_api_base_and_key_when_set(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(
            _settings(
                dim=4,
                api_base="https://openrouter.ai/api/v1",
                api_key="sk-test",
            ),
            embedding_fn=stub,
        )
        provider.embed_batch(["x"])
        call = stub.calls[0]
        assert call["api_base"] == "https://openrouter.ai/api/v1"
        assert call["api_key"] == "sk-test"
        assert call["model"] == "voyage/voyage-code-3"
        # Retry / timeout knobs make it through.
        assert call["timeout"] == 30.0
        assert call["num_retries"] == 3

    def test_omits_api_base_and_key_when_unset(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=4), embedding_fn=stub)
        provider.embed_batch(["x"])
        call = stub.calls[0]
        assert "api_base" not in call
        assert "api_key" not in call


# --------------------------------------------------------------------------- #
# Dimension safety                                                            #
# --------------------------------------------------------------------------- #


class TestDimValidation:
    def test_raises_when_returned_dim_does_not_match_settings(self) -> None:
        # Settings says dim=8 but the stub will return dim=4 vectors.
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=8), embedding_fn=stub)
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            provider.embed_batch(["abc"])

    def test_raises_when_sdk_returns_wrong_row_count(self) -> None:
        # Build a stub that drops one row to simulate provider misbehaviour.
        class _SkewedStub:
            def __call__(self, **kwargs: object) -> _StubResponse:
                inputs = kwargs["input"]
                assert isinstance(inputs, list)
                return _StubResponse(
                    data=[{"embedding": [1.0, 0.0, 0.0, 0.0], "index": 0} for _ in inputs[:-1]]
                )

        # batch_size=4 so all 3 inputs are sent in one call; the stub returns
        # only 2 vectors which must trigger the row-count guard.
        provider = LiteLLMEmbeddingProvider(
            _settings(dim=4, batch_size=4), embedding_fn=_SkewedStub()
        )
        with pytest.raises(RuntimeError, match="2 vectors for 3 inputs"):
            provider.embed_batch(["a", "b", "c"])


# --------------------------------------------------------------------------- #
# Pure helper                                                                 #
# --------------------------------------------------------------------------- #


class TestNormalise:
    def test_returns_unit_vector(self) -> None:
        v = _normalise([3.0, 4.0])  # 3-4-5 triangle → norm 5
        assert v == pytest.approx([0.6, 0.8])

    def test_zero_vector_is_left_alone(self) -> None:
        v = _normalise([0.0, 0.0, 0.0])
        assert v == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Payload-too-large detection + adaptive halving                              #
# --------------------------------------------------------------------------- #


class TestPayloadTooLargeDetection:
    @pytest.mark.parametrize(
        "msg",
        [
            "HTTP 413: Request Entity Too Large",
            "OpenrouterException - HTTP 400: 413 Request Entity Too Large",
            "payload too large for endpoint",
            "Request too large for context length",
            "max_tokens_per_request exceeded",
            # OpenRouter gateway-level 8 MB cap (returned as a 400, not a 413).
            'OpenrouterException - {"error":{"message":'
            '"The total text input size exceeds 8 MB","code":400}}',
            # Per-input character cap (HTTP 422 from Voyage / Qwen-3 / etc.)
            # — exactly the failure mode that motivated the per-input
            # pre-truncation + zero-vector-fallback fix.
            "BadRequestError: OpenrouterException - HTTP 422: "
            '{"error":{"message":"Value error, The input sequence should '
            'have less than 131072 characters. Input length: 10377148"}}',
        ],
    )
    def test_recognises_known_markers(self, msg: str) -> None:
        assert _is_payload_too_large(Exception(msg)) is True

    def test_unrelated_errors_pass_through(self) -> None:
        assert _is_payload_too_large(Exception("401 Unauthorized")) is False
        assert _is_payload_too_large(Exception("connection reset by peer")) is False


# --------------------------------------------------------------------------- #
# Per-input character cap (pre-truncation)                                    #
# --------------------------------------------------------------------------- #


class TestMaxInputCharsPreTruncation:
    def test_oversize_input_is_truncated_before_sdk_call(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(
            _settings(dim=4, batch_size=4, max_input_chars=100),
            embedding_fn=stub,
        )
        provider.embed_batch(["x" * 250, "ok"])
        # The SDK must see a truncated copy, never the original 250-char blob.
        sent = stub.calls[0]["input"]
        assert isinstance(sent, list)
        assert len(sent[0]) == 100
        assert sent[1] == "ok"

    def test_inputs_under_cap_pass_through_unchanged(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(
            _settings(dim=4, batch_size=4, max_input_chars=100),
            embedding_fn=stub,
        )
        provider.embed_batch(["short", "also short"])
        sent = stub.calls[0]["input"]
        assert sent == ["short", "also short"]

    def test_zero_or_negative_cap_disables_truncation(self) -> None:
        stub = _StubEmbed(dim=4)
        provider = LiteLLMEmbeddingProvider(
            _settings(dim=4, batch_size=4, max_input_chars=0),
            embedding_fn=stub,
        )
        provider.embed_batch(["x" * 5000])
        sent = stub.calls[0]["input"]
        assert isinstance(sent, list)
        assert len(sent[0]) == 5000


class _SizeCappedStub:
    """Stub that 413s any call whose batch exceeds ``max_batch``.

    Lets the halving recursion converge on a working batch size and tracks
    every (sub-)call so the test can assert tree shape.
    """

    def __init__(self, *, max_batch: int, dim: int) -> None:
        self.max_batch = max_batch
        self._dim = dim
        self.attempted_sizes: list[int] = []
        self.successful_sizes: list[int] = []

    def __call__(self, **kwargs: object) -> _StubResponse:
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        self.attempted_sizes.append(len(inputs))
        if len(inputs) > self.max_batch:
            raise RuntimeError("HTTP 413: Request Entity Too Large")
        self.successful_sizes.append(len(inputs))
        return _StubResponse(
            data=[
                {"embedding": [1.0, 0.0, 0.0, 0.0][: self._dim], "index": i}
                for i, _ in enumerate(inputs)
            ]
        )


class TestAdaptiveHalving:
    def test_halves_until_batch_fits(self) -> None:
        # batch_size=8 but the upstream only accepts 2-at-a-time.
        stub = _SizeCappedStub(max_batch=2, dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=4, batch_size=8), embedding_fn=stub)
        out = provider.embed_batch(["a", "b", "c", "d", "e", "f", "g", "h"])
        assert len(out) == 8
        # First attempt at 8 fails → halves into 4+4 (both fail) → 2+2+2+2 (all succeed).
        assert max(stub.attempted_sizes) == 8
        assert all(s <= 2 for s in stub.successful_sizes)
        assert sum(stub.successful_sizes) == 8

    def test_single_oversized_input_returns_zero_vector(self) -> None:
        """A single still-rejected input must NOT crash the run.

        One rogue file (auto-generated bundle, minified asset) used to abort
        the entire workspace embed when halving converged on a 1-element
        batch that the upstream still 413/422'd. We now emit a zero vector
        for that one chunk and keep going.
        """
        # max_batch=0 → every call 413s, even a 1-element batch.
        stub = _SizeCappedStub(max_batch=0, dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=4, batch_size=4), embedding_fn=stub)
        out = provider.embed_batch(["x"])
        assert len(out) == 1
        assert out[0] == [0.0, 0.0, 0.0, 0.0]

    def test_oversized_input_in_larger_batch_does_not_kill_siblings(self) -> None:
        """Mixed batch: only the truly oversized input gets a zero vector.

        The halving recursion must isolate the failing input — its siblings
        should still receive real embeddings on subsequent halves.
        """

        class _OnePoisonStub:
            """Reject any batch that includes the input ``"POISON"``; succeed otherwise."""

            def __init__(self, dim: int) -> None:
                self._dim = dim
                self.calls: list[int] = []

            def __call__(self, **kwargs: object) -> _StubResponse:
                inputs = kwargs["input"]
                assert isinstance(inputs, list)
                self.calls.append(len(inputs))
                if "POISON" in inputs:
                    raise RuntimeError("HTTP 413: Request Entity Too Large")
                return _StubResponse(
                    data=[
                        {"embedding": [1.0, 0.0, 0.0, 0.0][: self._dim], "index": i}
                        for i, _ in enumerate(inputs)
                    ]
                )

        stub = _OnePoisonStub(dim=4)
        provider = LiteLLMEmbeddingProvider(_settings(dim=4, batch_size=4), embedding_fn=stub)
        out = provider.embed_batch(["a", "b", "POISON", "c"])
        assert len(out) == 4
        # Position 2 is the poison input → zero vector. The others are
        # successfully embedded and L2-normalised (so non-zero).
        assert out[2] == [0.0, 0.0, 0.0, 0.0]
        for i in (0, 1, 3):
            assert any(x != 0.0 for x in out[i]), f"sibling at index {i} lost its embedding"

    def test_non_413_errors_are_not_halved(self) -> None:
        # Auth errors must propagate immediately — halving wouldn't help and
        # would just hammer the upstream with duplicate failed calls.
        call_count = {"n": 0}

        def _auth_fail(**_: object) -> _StubResponse:
            call_count["n"] += 1
            raise RuntimeError("401 Unauthorized: bad API key")

        provider = LiteLLMEmbeddingProvider(_settings(dim=4, batch_size=4), embedding_fn=_auth_fail)
        with pytest.raises(RuntimeError, match="401"):
            provider.embed_batch(["a", "b", "c", "d"])
        assert call_count["n"] == 1, "auth errors must not trigger batch halving"
