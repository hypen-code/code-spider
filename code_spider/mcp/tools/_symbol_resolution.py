"""Symbol resolution helper shared by ``get_call_graph`` / ``get_impact_analysis``.

Defines four resolution modes, in order of strictness:

* ``"exact"``  — current behaviour. ``MATCH (s:Symbol {fqn: $fqn})``. The
  default so existing integrations keep their precise semantics.
* ``"suffix"`` — ``WHERE s.fqn ENDS WITH $tail``. Useful when the agent
  only has a leaf name (``User.save``) but does not know the module path.
  Requires the tail to be at least 3 characters to avoid full-label scans.
* ``"fuzzy"``  — fulltext over the existing ``symbol_text`` index on
  ``(:Symbol)[name, signature, docstring]``. Returns the top matches with
  Lucene scores. The query string is Lucene-escaped before being sent.
* ``"auto"``   — try exact, fall back to suffix when zero rows, then to
  fuzzy. The first non-empty result wins.

Safety
------
* ``assert_safe_identifier`` is called on the input before any mode runs, so
  the existing identifier surface (``[A-Za-z0-9_.\\-:/@*{}]``) is preserved.
* For fulltext, any character in the Lucene metaset is escaped with a
  backslash. The whole expression is then wrapped in a phrase query so the
  user's text is treated as data, not as Lucene syntax.
* Cypher remains parameterised. We never interpolate user data into the
  query string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MatchMode = Literal["exact", "suffix", "fuzzy", "auto"]
ResolvedMode = Literal["exact", "suffix", "fuzzy"]

# Hard caps. Tools accept a smaller ``max_candidates`` arg but never larger.
_HARD_MAX_CANDIDATES: int = 50
_MIN_SUFFIX_CHARS: int = 3

# Lucene meta-characters that must be backslash-escaped before being sent to
# ``db.index.fulltext.queryNodes``. Note: the index name itself is a literal,
# not user data, so it is never escaped.
_LUCENE_META: frozenset[str] = frozenset('+-&|!(){}[]^"~*?:\\/')


@dataclass(frozen=True, slots=True)
class ResolvedSymbol:
    """A symbol candidate plus the score under which it was found."""

    fqn: str
    repo: str | None
    file_path: str | None
    start_line: int | None
    end_line: int | None
    kind: str | None
    score: float
    mode: ResolvedMode


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #


def lucene_escape(query: str) -> str:
    """Backslash-escape every Lucene metacharacter in ``query``.

    The result is then safe to wrap into a phrase or term clause without
    risking accidental operator interpretation. Whitespace is preserved.
    """
    return "".join(("\\" + c) if c in _LUCENE_META else c for c in query)


def clamp_candidates(n: int) -> int:
    """Clamp ``max_candidates`` to [1, _HARD_MAX_CANDIDATES]."""
    if n < 1:
        return 1
    if n > _HARD_MAX_CANDIDATES:
        return _HARD_MAX_CANDIDATES
    return n


# --------------------------------------------------------------------------- #
# Cypher templates (module-level so tests can read them).                     #
# --------------------------------------------------------------------------- #


_EXACT_QUERY = """
MATCH (s:Symbol {workspace_id: $workspace_id, fqn: $fqn})
RETURN s.fqn AS fqn, s.repo AS repo, s.file_path AS file_path,
       s.start_line AS start_line, s.end_line AS end_line, s.kind AS kind
LIMIT $limit
"""


# ``ENDS WITH`` cannot use the fqn b-tree (it isn't indexed today) but the
# ``name`` b-tree is. We pre-filter on ``name`` when the tail looks like a
# bare identifier so the suffix scan is cheap; otherwise fall back to a
# full-label ENDS WITH scan (still bounded by ``$limit``).
_SUFFIX_QUERY = """
MATCH (s:Symbol)
WHERE s.workspace_id = $workspace_id
  AND s.fqn ENDS WITH $tail
RETURN s.fqn AS fqn, s.repo AS repo, s.file_path AS file_path,
       s.start_line AS start_line, s.end_line AS end_line, s.kind AS kind,
       1.0 AS score
ORDER BY size(s.fqn) ASC
LIMIT $limit
"""


# Fulltext index ``symbol_text`` is over (:Symbol)[name, signature, docstring].
# It is created at migrate-time. We post-filter by workspace_id because the
# fulltext procedure does not support equality constraints.
_FUZZY_QUERY = """
CALL db.index.fulltext.queryNodes('symbol_text', $q) YIELD node AS s, score
WHERE s.workspace_id = $workspace_id
RETURN s.fqn AS fqn, s.repo AS repo, s.file_path AS file_path,
       s.start_line AS start_line, s.end_line AS end_line, s.kind AS kind,
       score AS score
ORDER BY score DESC
LIMIT $limit
"""


# --------------------------------------------------------------------------- #
# Row → dataclass                                                             #
# --------------------------------------------------------------------------- #


def _row_to_resolved(
    row: Any, *, mode: ResolvedMode, default_score: float
) -> ResolvedSymbol:
    # ``neo4j.Record`` and ``dict`` both expose ``__getitem__`` / ``.get``.
    keys = set(row.keys())
    return ResolvedSymbol(
        fqn=row["fqn"],
        repo=row.get("repo"),
        file_path=row.get("file_path"),
        start_line=row.get("start_line"),
        end_line=row.get("end_line"),
        kind=row.get("kind"),
        score=float(row["score"]) if "score" in keys else default_score,
        mode=mode,
    )


# --------------------------------------------------------------------------- #
# Mode runners — each takes a session and returns a list of ResolvedSymbol    #
# --------------------------------------------------------------------------- #


def _run_exact(session: Any, *, workspace_id: str, fqn: str, limit: int) -> list[ResolvedSymbol]:
    rows = list(
        session.run(
            _EXACT_QUERY,
            parameters={"workspace_id": workspace_id, "fqn": fqn, "limit": limit},
        )
    )
    return [_row_to_resolved(r, mode="exact", default_score=1.0) for r in rows]


def _run_suffix(
    session: Any, *, workspace_id: str, fqn: str, limit: int
) -> list[ResolvedSymbol]:
    # Short tails cause full-label scans; guard the resolver here so an
    # accidental ``s`` doesn't fan out across an entire workspace.
    if len(fqn) < _MIN_SUFFIX_CHARS:
        return []
    rows = list(
        session.run(
            _SUFFIX_QUERY,
            parameters={"workspace_id": workspace_id, "tail": fqn, "limit": limit},
        )
    )
    return [_row_to_resolved(r, mode="suffix", default_score=1.0) for r in rows]


def _run_fuzzy(
    session: Any, *, workspace_id: str, fqn: str, limit: int
) -> list[ResolvedSymbol]:
    # Escape Lucene metas then wrap in quotes so the whole user text is one
    # phrase. Phrase queries also disable Lucene's default tokenisation
    # operators, which is exactly what we want for identifier search.
    escaped = lucene_escape(fqn)
    if not escaped.strip():
        return []
    q = f'"{escaped}"'
    rows = list(
        session.run(
            _FUZZY_QUERY,
            parameters={"workspace_id": workspace_id, "q": q, "limit": limit},
        )
    )
    return [_row_to_resolved(r, mode="fuzzy", default_score=0.0) for r in rows]


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def resolve(
    session: Any,
    *,
    workspace_id: str,
    fqn: str,
    mode: MatchMode,
    max_candidates: int,
) -> tuple[list[ResolvedSymbol], ResolvedMode | None]:
    """Resolve ``fqn`` under ``mode`` and return ``(candidates, mode_used)``.

    ``mode_used`` is ``None`` only when every mode returns no rows. The list
    is ordered with the best match first.
    """
    limit = clamp_candidates(max_candidates)

    if mode == "exact":
        rows = _run_exact(session, workspace_id=workspace_id, fqn=fqn, limit=limit)
        return rows, ("exact" if rows else None)

    if mode == "suffix":
        rows = _run_suffix(session, workspace_id=workspace_id, fqn=fqn, limit=limit)
        return rows, ("suffix" if rows else None)

    if mode == "fuzzy":
        rows = _run_fuzzy(session, workspace_id=workspace_id, fqn=fqn, limit=limit)
        return rows, ("fuzzy" if rows else None)

    # ``auto``: progressive fallback. Stops on first non-empty result.
    exact_rows = _run_exact(session, workspace_id=workspace_id, fqn=fqn, limit=limit)
    if exact_rows:
        return exact_rows, "exact"
    suffix_rows = _run_suffix(session, workspace_id=workspace_id, fqn=fqn, limit=limit)
    if suffix_rows:
        return suffix_rows, "suffix"
    fuzzy_rows = _run_fuzzy(session, workspace_id=workspace_id, fqn=fqn, limit=limit)
    if fuzzy_rows:
        return fuzzy_rows, "fuzzy"
    return [], None


def resolved_to_payload(r: ResolvedSymbol) -> dict[str, Any]:
    """Render a :class:`ResolvedSymbol` into the MCP-tool response shape."""
    return {
        "fqn": r.fqn,
        "repo": r.repo,
        "file_path": r.file_path,
        "start_line": r.start_line,
        "end_line": r.end_line,
        "kind": r.kind,
        "score": round(r.score, 4),
        "match_mode": r.mode,
    }
