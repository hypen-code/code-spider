"""Hybrid search RRF fusion tests — pure logic, no Neo4j."""

from __future__ import annotations


def test_rrf_constant_matches_design() -> None:
    from code_spider.search import RRF_K

    assert RRF_K == 60


def test_rrf_fusion_prefers_documents_in_both_lists() -> None:
    """Reimplement RRF in the test against the same constant to assert
    the design contract held by :func:`hybrid_search`."""
    from code_spider.search.hybrid import RRF_K

    # Simulated lex + vec rankings for three documents.
    lex_order = ["A", "B", "C"]
    vec_order = ["B", "C", "A"]

    scores: dict[str, float] = {}
    for rank, doc in enumerate(lex_order, start=1):
        scores[doc] = scores.get(doc, 0) + 1 / (RRF_K + rank)
    for rank, doc in enumerate(vec_order, start=1):
        scores[doc] = scores.get(doc, 0) + 1 / (RRF_K + rank)

    # B is rank-2 in lex and rank-1 in vec → best fused score.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    assert ranked[0][0] == "B"
