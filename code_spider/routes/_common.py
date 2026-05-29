"""Shared helpers for route + HTTP client extraction across languages."""

from __future__ import annotations

import re

_VALID_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "CONNECT", "TRACE"}
)

# Normalise framework-specific path-parameter syntax to ``{}`` placeholders.
# Examples:
#   /users/{id}         (FastAPI)      -> /users/{}
#   /users/{id:int}     (FastAPI typed)-> /users/{}
#   /users/:id          (Express)      -> /users/{}
#   /users/[id]         (Next.js Pages)-> /users/{}
#   /users/<int:id>     (Flask/Django) -> /users/{}
_PARAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\{[^/{}]+\}"),
    re.compile(r":[A-Za-z_][A-Za-z0-9_]*"),
    re.compile(r"\[[^/\[\]]+\]"),
    re.compile(r"<[^/<>]+>"),
)


def is_http_method(name: str) -> bool:
    return name.upper() in _VALID_METHODS


def normalize_method(name: str) -> str:
    return name.upper()


def normalize_path(path: str) -> str:
    """Collapse framework-specific path-param syntax to ``{}`` placeholders."""
    if not path:
        return path
    normalised = path
    for pat in _PARAM_PATTERNS:
        normalised = pat.sub("{}", normalised)
    # Always lead with '/'.
    if not normalised.startswith("/"):
        normalised = "/" + normalised
    # Collapse duplicate slashes.
    normalised = re.sub(r"/+", "/", normalised)
    # Drop trailing slash (but never produce empty string).
    if len(normalised) > 1 and normalised.endswith("/"):
        normalised = normalised[:-1]
    return normalised


def path_segments(path: str) -> list[str]:
    """Split a normalised path into non-empty segments."""
    return [s for s in path.split("/") if s]


def path_similarity(a: str, b: str) -> float:
    """Jaccard similarity over path segments, with a positional bonus.

    Returns a score in ``[0.0, 1.0]``. A perfect string match yields 1.0;
    completely disjoint paths yield 0.0.
    """
    na, nb = normalize_path(a), normalize_path(b)
    if na == nb:
        return 1.0
    sa = path_segments(na)
    sb = path_segments(nb)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    if len(sa) != len(sb):
        # Different number of segments cannot be the same route.
        return 0.0
    matched = sum(
        1
        for x, y in zip(sa, sb, strict=False)
        if x == y or "{}" in (x, y)
    )
    return matched / len(sa)
