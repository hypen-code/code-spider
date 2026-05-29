"""Call resolver — 6-strategy heuristic cascade.

Strategies in priority order:

    1. Same-module lookup            (confidence 0.95)
    2. Import-map suffix fallback    (confidence 0.90)
    3. Unique-name global lookup     (confidence 0.85)
    4. Suffix matching by distance   (confidence 0.75)
    5. Syntactic suffix fallback     (confidence 0.55)
    6. Unresolved text-only          (confidence 0.30 — no edge emitted)

Public entry point: :func:`resolve_workspace`.
"""

from code_spider.resolver.cascade import resolve_workspace
from code_spider.resolver.index import ImportMap, SymbolIndex, SymbolRef, build_indexes

__all__ = [
    "ImportMap",
    "SymbolIndex",
    "SymbolRef",
    "build_indexes",
    "resolve_workspace",
]
