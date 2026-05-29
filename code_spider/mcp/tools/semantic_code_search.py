"""``semantic_code_search`` — hybrid lexical + vector RRF search."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from code_spider.mcp.auth import assert_safe_workspace_id, audited
from code_spider.mcp.context import get_context
from code_spider.search.hybrid import hybrid_search


@audited("semantic_code_search")
def semantic_code_search(
    workspace_id: str,
    query: str,
    limit: int = 10,
    lexical_k: int = 30,
    vector_k: int = 30,
) -> dict[str, Any]:
    """Return ranked code hits combining symbol fulltext + chunk vector search.

    Lexical and vector results are fused with Reciprocal Rank Fusion (k=60).
    Each result carries ``file_path``, ``start_line``, ``end_line``,
    ``rrf_score`` and the ``sources`` that contributed.

    Tune ``lexical_k`` / ``vector_k`` to widen the candidate pool before
    fusion (higher = more recall, slower).
    """
    assert_safe_workspace_id(workspace_id)
    if not query or not query.strip():
        return {"hits": [], "lexical_k": lexical_k, "vector_k": vector_k}
    if limit < 1 or limit > 100:
        raise ValueError("limit must be in [1, 100]")
    if lexical_k < 0 or vector_k < 0 or lexical_k + vector_k == 0:
        raise ValueError("lexical_k + vector_k must be > 0")

    ctx = get_context()
    hits = hybrid_search(
        client=ctx.neo4j,
        backend=ctx.vector,
        provider=ctx.embedder,
        workspace_id=workspace_id,
        query=query,
        limit=limit,
        lexical_k=lexical_k,
        vector_k=vector_k,
    )
    return {
        "hits": [_dict(h) for h in hits],
        "lexical_k": lexical_k,
        "vector_k": vector_k,
    }


def _dict(hit) -> dict[str, Any]:
    d = asdict(hit)
    d["sources"] = list(d["sources"])
    return d
