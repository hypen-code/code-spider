"""``sentence-transformers`` adapter implementing :class:`EmbeddingProvider`.

Loads ``all-MiniLM-L6-v2`` (384-dim) lazily so tests/builds that never call
:meth:`embed_batch` do not pay the model download/load cost. Uses CUDA when
available; falls back to CPU otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_spider.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

_log = get_logger(__name__)


class SentenceTransformerProvider:
    """Production embedding provider using ``sentence-transformers``."""

    name: str = "sentence-transformers"
    dim: int = 384

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for the default embedding "
                    "provider. Install with: pip install 'code-spider[embedding]'"
                ) from exc
            _log.info("loading embedding model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in vectors]
