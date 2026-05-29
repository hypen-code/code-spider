"""LiteLLM-backed embedding provider.

LiteLLM is a thin SDK that translates a single ``embedding(...)`` call into
the right native API call for ~100 providers (OpenAI, Voyage, Cohere,
Mistral, Vertex AI, Bedrock, Azure, etc.) so we don't have to ship a
per-provider adapter for each.

Recommended models for **code retrieval** (best to most general):

* ``voyage/voyage-code-3`` — Voyage AI, 1024 dim, tuned on code; tops most
  code-retrieval leaderboards in 2025. Requires ``VOYAGE_API_KEY``.
* ``openai/text-embedding-3-small`` — 1536 dim, cheap, broadly available.
  Requires ``OPENAI_API_KEY``.
* ``cohere/embed-multilingual-v3.0`` — 1024 dim, strong on mixed corpora.
  Requires ``COHERE_API_KEY``.

``OpenRouter`` currently exposes chat completions but not embeddings; if/when
they add an ``/embeddings`` endpoint, point ``CODE_SPIDER_EMBED_API_BASE`` at
``https://openrouter.ai/api/v1`` and supply ``OPENROUTER_API_KEY``. LiteLLM
will route through it because it speaks the OpenAI wire protocol.

Design highlights:

* **Lazy import** of ``litellm`` so installs without the optional dep do not
  pay the import cost (LiteLLM pulls a fair amount of code).
* **Batched HTTP** — every ``embed_batch`` chunks the inputs into
  ``batch_size`` slices and issues one provider call per slice.
* **Retries with backoff** via the SDK's built-in ``num_retries`` knob.
* **Strict dim validation** — the first response defines an expected
  dimension that must match ``settings.embedding.dim``; subsequent
  responses are checked too. A mismatch raises immediately because the
  Neo4j vector index is dimension-pinned.
* **L2 normalisation** — done locally so downstream cosine similarity is
  uniform regardless of whether the upstream provider already normalised.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from code_spider.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from code_spider.config import EmbeddingSettings

_log = get_logger(__name__)


def _normalise(vec: list[float]) -> list[float]:
    """L2-normalise ``vec`` so cosine similarity is a dot product."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


# Substrings that identify "shrink the batch" errors. The upstream gateway
# (OpenRouter / Cloudflare / nginx in front of a provider) returns these when
# the request body exceeds its per-call cap; the right response is to halve
# the batch and retry rather than fail the whole indexing run.
_PAYLOAD_TOO_LARGE_MARKERS: tuple[str, ...] = (
    "413",
    "request entity too large",
    "payload too large",
    "request too large",
    "max_tokens_per_request",
    "context length",
    # OpenRouter's gateway-level size cap surfaces as a 400 with this exact
    # phrasing instead of a 413; treat it as "shrink and retry" too.
    "input size exceeds",
    "total text input size",
    # Per-input character cap (Voyage / Qwen-3 / OpenAI larger context models):
    #     "Value error, The input sequence should have less than 131072 characters."
    # When the single offending input is still over the per-request cap after
    # halving down to size 1, the provider returns a 422 with this wording.
    "input sequence should have less than",
    "input length",
)


def _is_payload_too_large(exc: BaseException) -> bool:
    """Best-effort detection of a "batch too big" error from any provider.

    We don't import LiteLLM exception types directly to keep this file
    importable without the optional dep; matching on the stringified message
    works across providers (OpenAI, OpenRouter, Voyage, Cohere all surface a
    recognisable substring).
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _PAYLOAD_TOO_LARGE_MARKERS)


def _quiet_litellm_logging() -> None:
    """Silence LiteLLM's per-call INFO chatter + "Provider List" prints.

    LiteLLM logs every successful call at INFO and prints a verbose footer to
    stderr on errors. Both are useless inside the indexer's structured-log
    output and visually drown the real error. ``suppress_debug_info`` kills
    the prints; bumping the ``LiteLLM`` logger to WARNING kills the per-call
    INFO lines. Safe to call repeatedly.
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover — caller already handled this
        return
    litellm.suppress_debug_info = True
    # Cover both casings seen across LiteLLM versions.
    for name in ("LiteLLM", "litellm"):
        logging.getLogger(name).setLevel(logging.WARNING)


class LiteLLMEmbeddingProvider:
    """Embedding provider that delegates to the LiteLLM SDK.

    Attributes:
        name: Always ``"litellm"`` so the provider registry can resolve it.
        dim: The expected embedding dimension; must match
            ``settings.embedding.dim`` and the Neo4j vector index.
    """

    name: str = "litellm"

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        # Hook for unit tests — inject a fake ``litellm.embedding`` callable
        # without going through the global SDK.
        embedding_fn: Any = None,
    ) -> None:
        if not settings.model:
            raise ValueError(
                "CODE_SPIDER_EMBED_MODEL is required for the litellm provider "
                "(e.g. 'voyage/voyage-code-3', 'openai/text-embedding-3-small')."
            )
        self._settings = settings
        self._batch_size = max(1, settings.batch_size)
        self._embedding_fn = embedding_fn  # resolved lazily when None

    @property
    def dim(self) -> int:
        return self._settings.dim

    # --- Internals --------------------------------------------------------- #

    def _resolve_sdk(self) -> Any:
        """Lazily import :mod:`litellm`. Cached on the instance."""
        if self._embedding_fn is not None:
            return self._embedding_fn
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required for the 'litellm' embedding provider. "
                "Install with: pip install 'code-spider[litellm]'"
            ) from exc
        # Mute LiteLLM's per-call INFO logs + "Provider List" / "Give Feedback"
        # prints so the indexer's structured log output stays readable.
        _quiet_litellm_logging()
        self._embedding_fn = litellm.embedding
        return self._embedding_fn

    def _request_kwargs(self, inputs: list[str]) -> dict[str, Any]:
        """Build the kwargs dict passed to ``litellm.embedding``."""
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "input": inputs,
            "timeout": self._settings.timeout_s,
            "num_retries": self._settings.max_retries,
        }
        if self._settings.api_base:
            kwargs["api_base"] = self._settings.api_base
        if self._settings.api_key:
            kwargs["api_key"] = self._settings.api_key
        return kwargs

    @staticmethod
    def _extract_vectors(response: Any) -> list[list[float]]:
        """Pull ``data[i].embedding`` out of a LiteLLM response.

        LiteLLM returns an ``EmbeddingResponse`` whose ``data`` is a list of
        dicts or ``Embedding`` objects (depending on version). Both expose
        ``embedding``; we handle both shapes.
        """
        data = response["data"] if isinstance(response, dict) else response.data
        out: list[list[float]] = []
        for item in data:
            vec = item["embedding"] if isinstance(item, dict) else item.embedding
            out.append([float(x) for x in vec])
        return out

    def _embed_one_call(self, embed: Any, chunk: list[str]) -> list[list[float]]:
        """Issue a single ``litellm.embedding`` call and return raw vectors.

        Wraps the kwargs build + extraction so the caller can focus on
        retry / halving policy rather than transport plumbing.
        """
        kwargs = self._request_kwargs(chunk)
        response = embed(**kwargs)
        vectors = self._extract_vectors(response)
        if len(vectors) != len(chunk):
            raise RuntimeError(f"litellm returned {len(vectors)} vectors for {len(chunk)} inputs")
        return vectors

    def _embed_with_halving(self, embed: Any, chunk: list[str]) -> list[list[float]]:
        """Embed ``chunk``, halving and recursing on payload-too-large errors.

        Some gateways (notably OpenRouter's nginx layer in front of certain
        backends like Perplexity) impose a per-request body cap well below
        the provider's documented batch limit. When we see a 413/422 we split
        the batch in half and try each half.

        Bottoming out at a *single* still-oversized input is the failure mode
        a giant auto-generated or minified source file triggers (e.g. a 10 MB
        bundle that slips past ``max_input_chars`` pre-truncation because the
        cap was misconfigured). To prevent one such input from killing the
        whole workspace embed, we log a clear warning and substitute a
        zero-vector for that one chunk; downstream cosine similarity against
        a zero vector is always 0 so the chunk is effectively excluded from
        vector search without poisoning anything else.
        """
        try:
            return self._embed_one_call(embed, chunk)
        except Exception as exc:
            if _is_payload_too_large(exc):
                if len(chunk) > 1:
                    mid = len(chunk) // 2
                    _log.warning(
                        "litellm payload too large; halving batch and retrying",
                        model=self._settings.model,
                        from_size=len(chunk),
                        to_size=mid,
                    )
                    left = self._embed_with_halving(embed, chunk[:mid])
                    right = self._embed_with_halving(embed, chunk[mid:])
                    return left + right
                # Single input still rejected — skip it instead of crashing
                # the whole run. The zero vector is a sentinel: it is not
                # L2-normalised (`_normalise` returns zero-vectors unchanged)
                # so cosine similarity against it is always 0.
                _log.error(
                    "litellm rejected single oversized input; emitting zero "
                    "vector to skip without aborting the workspace embed",
                    model=self._settings.model,
                    input_chars=len(chunk[0]) if chunk else 0,
                    max_input_chars=self._settings.max_input_chars,
                    error=str(exc),
                )
                return [[0.0] * self._settings.dim]
            _log.error(
                "litellm embedding call failed",
                model=self._settings.model,
                api_base=self._settings.api_base,
                batch_size=len(chunk),
                error=str(exc),
            )
            raise

    def _truncate_oversize(self, texts: list[str]) -> list[str]:
        """Pre-truncate any single input longer than ``max_input_chars``.

        The hosted embedding endpoints we target all enforce a per-input
        character (not just per-batch) cap and reject the entire request when
        any one input exceeds it. Truncating to a safe ceiling at the
        provider boundary is the cheapest way to keep one rogue file from
        crashing the workspace run. The truncated chunk still gets *an*
        embedding — partial vector coverage of a 10 MB blob is rarely useful,
        but it's better than aborting and losing every other chunk in the
        same indexing run.
        """
        cap = self._settings.max_input_chars
        if cap <= 0:
            return texts
        out: list[str] = []
        oversize = 0
        worst = 0
        for t in texts:
            if len(t) > cap:
                oversize += 1
                worst = max(worst, len(t))
                out.append(t[:cap])
            else:
                out.append(t)
        if oversize:
            _log.warning(
                "pre-truncated oversize inputs before embedding",
                model=self._settings.model,
                oversize_count=oversize,
                worst_chars=worst,
                cap_chars=cap,
            )
        return out

    # --- Public API -------------------------------------------------------- #

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in provider-bounded batches; returns L2-normalised vectors.

        Inputs longer than ``max_input_chars`` are pre-truncated at the
        provider boundary so a single rogue file cannot crash the run. If
        the upstream still rejects a single oversized chunk (misconfigured
        cap, unusual model limit), that chunk gets a zero-vector and a loud
        warning instead of aborting the whole call.
        """
        if not texts:
            return []
        embed = self._resolve_sdk()
        out: list[list[float]] = []
        expected_dim = self._settings.dim

        safe_texts = self._truncate_oversize(texts)

        for start in range(0, len(safe_texts), self._batch_size):
            chunk = safe_texts[start : start + self._batch_size]
            vectors = self._embed_with_halving(embed, chunk)
            for v in vectors:
                if len(v) != expected_dim:
                    raise RuntimeError(
                        "embedding dimension mismatch: model "
                        f"{self._settings.model!r} returned {len(v)} dims but "
                        f"CODE_SPIDER_EMBED_DIM is {expected_dim}. Re-run "
                        "`code-spider migrate` after updating CODE_SPIDER_EMBED_DIM "
                        "to match your chosen model."
                    )
                out.append(_normalise(v))
        return out
