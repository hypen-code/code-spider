"""Vector ANN search over the ``chunk_embedding`` index."""

from __future__ import annotations

from code_spider.embedding.provider import EmbeddingProvider
from code_spider.graph.backends import VectorHit
from code_spider.graph.neo4j_vector import Neo4jVectorBackend


def vector_search(
    *,
    backend: Neo4jVectorBackend,
    provider: EmbeddingProvider,
    workspace_id: str,
    query: str,
    limit: int = 20,
) -> list[VectorHit]:
    """Embed ``query`` and return the closest ``limit`` chunks in ``workspace_id``."""
    if not query.strip():
        return []
    embedding = provider.embed_batch([query])[0]
    return backend.query(workspace_id=workspace_id, embedding=embedding, limit=limit)
