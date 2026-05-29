"""Hybrid search: Neo4j fulltext + native vector ANN, fused via Reciprocal Rank Fusion."""

from code_spider.search.hybrid import RRF_K, HybridHit, hybrid_search
from code_spider.search.lexical import LexicalHit, lexical_search
from code_spider.search.vector import vector_search

__all__ = [
    "RRF_K",
    "HybridHit",
    "LexicalHit",
    "hybrid_search",
    "lexical_search",
    "vector_search",
]
