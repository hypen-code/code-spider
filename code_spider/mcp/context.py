"""Lifecycle-managed context for the MCP server.

The server holds long-lived resources (Neo4j driver, embedding model, vector
backend) on a single :class:`ServerContext`. Tools fetch it via
:func:`get_context` rather than rebuilding connections per request.

Initialisation is explicit (``initialize`` at server startup) so tests can
inject fakes.
"""

from __future__ import annotations

from dataclasses import dataclass

from code_spider.config import Settings, load_settings
from code_spider.embedding.provider import EmbeddingProvider, get_embedding_provider
from code_spider.graph.client import Neo4jClient
from code_spider.graph.neo4j_vector import Neo4jVectorBackend
from code_spider.logging_setup import configure_logging, get_logger

_log = get_logger(__name__)


@dataclass(slots=True)
class ServerContext:
    """Long-lived MCP server resources."""

    settings: Settings
    neo4j: Neo4jClient
    vector: Neo4jVectorBackend
    embedder: EmbeddingProvider

    def close(self) -> None:
        self.neo4j.close()


_CONTEXT: ServerContext | None = None


def initialize(
    *,
    settings: Settings | None = None,
    embed_provider: str = "auto",
) -> ServerContext:
    """Build the shared :class:`ServerContext`. Call once at startup."""
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT

    s = settings or load_settings()
    configure_logging(s.log_level, json_output=s.log_json)
    client = Neo4jClient(s.neo4j)
    client.verify()
    vector = Neo4jVectorBackend(client)
    embedder = _resolve_embedder(embed_provider)
    _CONTEXT = ServerContext(
        settings=s,
        neo4j=client,
        vector=vector,
        embedder=embedder,
    )
    _log.info(
        "mcp server initialised",
        embed_provider=embedder.name,
        embed_dim=embedder.dim,
    )
    return _CONTEXT


def get_context() -> ServerContext:
    if _CONTEXT is None:
        raise RuntimeError(
            "ServerContext not initialised — call initialize() before starting tools."
        )
    return _CONTEXT


def shutdown() -> None:
    global _CONTEXT
    if _CONTEXT is not None:
        _CONTEXT.close()
        _CONTEXT = None


def _resolve_embedder(name: str) -> EmbeddingProvider:
    if name in ("auto", "sentence-transformers"):
        try:
            return get_embedding_provider("sentence-transformers")
        except (ImportError, KeyError):
            _log.warning(
                "sentence-transformers unavailable; MCP server falling back to "
                "deterministic hash embeddings (semantic_code_search quality "
                "will be poor). Install with: pip install 'code-spider[embedding]'"
            )
            return get_embedding_provider("hash")
    return get_embedding_provider(name)
