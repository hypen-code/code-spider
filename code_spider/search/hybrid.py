"""Hybrid search — fuse lexical (symbol fulltext) + vector (chunk ANN) via RRF.

Reciprocal Rank Fusion (Cormack et al., 2009) combines ranked lists from
heterogeneous retrievers without requiring score normalisation:

    rrf_score(doc) = sum over retrievers L of  1 / (k + rank_L(doc))

We use the standard ``k=60``.

Each retriever returns rows keyed on different node types (``Symbol`` vs
``Chunk``); the fuser normalises both into a unified :class:`HybridHit`.
"""

from __future__ import annotations

from dataclasses import dataclass

from code_spider.embedding.provider import EmbeddingProvider
from code_spider.graph.client import Neo4jClient
from code_spider.graph.neo4j_vector import Neo4jVectorBackend
from code_spider.search.lexical import LexicalHit, lexical_search
from code_spider.search.vector import vector_search

RRF_K = 60


@dataclass(frozen=True, slots=True)
class HybridHit:
    workspace_id: str
    repo: str | None
    fqn: str | None
    name: str | None
    kind: str  # "Symbol" or "Chunk"
    file_path: str
    start_line: int
    end_line: int
    rrf_score: float
    sources: tuple[str, ...]  # which retrievers contributed
    chunk_id: str | None = None


def hybrid_search(
    *,
    client: Neo4jClient,
    backend: Neo4jVectorBackend,
    provider: EmbeddingProvider,
    workspace_id: str,
    query: str,
    limit: int = 10,
    lexical_k: int = 30,
    vector_k: int = 30,
) -> list[HybridHit]:
    """Return the top ``limit`` hits fused from both retrievers."""
    lex_hits = lexical_search(
        client=client, workspace_id=workspace_id, query=query, limit=lexical_k
    )
    vec_hits = vector_search(
        backend=backend, provider=provider, workspace_id=workspace_id, query=query, limit=vector_k
    )

    rrf: dict[str, dict] = {}

    for rank, h in enumerate(lex_hits, start=1):
        key = f"sym::{h.fqn}"
        entry = rrf.setdefault(
            key,
            {
                "rrf": 0.0,
                "sources": set(),
                "kind": "Symbol",
                "data": h,
            },
        )
        entry["rrf"] += 1.0 / (RRF_K + rank)
        entry["sources"].add("lexical")

    for rank, h in enumerate(vec_hits, start=1):
        key = f"chk::{h.chunk_id}"
        entry = rrf.setdefault(
            key,
            {
                "rrf": 0.0,
                "sources": set(),
                "kind": "Chunk",
                "data": h,
            },
        )
        entry["rrf"] += 1.0 / (RRF_K + rank)
        entry["sources"].add("vector")

    fused = sorted(rrf.values(), key=lambda x: x["rrf"], reverse=True)[:limit]
    return [_to_hybrid_hit(entry, workspace_id) for entry in fused]


def _to_hybrid_hit(entry: dict, workspace_id: str) -> HybridHit:
    sources = tuple(sorted(entry["sources"]))
    data = entry["data"]
    if entry["kind"] == "Symbol":
        h: LexicalHit = data
        return HybridHit(
            workspace_id=h.workspace_id,
            repo=h.repo,
            fqn=h.fqn,
            name=h.name,
            kind="Symbol",
            file_path=h.file_path,
            start_line=h.start_line,
            end_line=h.end_line,
            rrf_score=round(entry["rrf"], 6),
            sources=sources,
        )
    # Chunk
    return HybridHit(
        workspace_id=workspace_id,
        repo=None,
        fqn=None,
        name=None,
        kind="Chunk",
        file_path=data.file_path,
        start_line=data.start_line,
        end_line=data.end_line,
        rrf_score=round(entry["rrf"], 6),
        sources=sources,
        chunk_id=data.chunk_id,
    )
