"""``get_call_graph`` MCP tool — callers + callees of a symbol via :CALLS edges."""

from __future__ import annotations

from typing import Any, Literal

from code_spider.mcp.auth import (
    assert_safe_identifier,
    assert_safe_search_text,
    assert_safe_workspace_id,
    audited,
    read_session,
)
from code_spider.mcp.context import get_context
from code_spider.mcp.tools._symbol_resolution import (
    MatchMode,
    clamp_candidates,
    resolve,
    resolved_to_payload,
)

# NOTE: ``c1`` and ``c2`` are *relationship lists* bound by the variable-length
# pattern ``-[var:CALLS*1..N]->``. In Cypher, ``length()`` only accepts a Path;
# for a list of relationships you must use ``size()``. Using ``length()`` here
# raised: "Type mismatch: expected Path but was List<Relationship>".
_QUERY = """
MATCH (target:Symbol {workspace_id: $workspace_id, fqn: $fqn})
OPTIONAL MATCH (caller:Symbol)-[c1:CALLS*1..%(depth)d]->(target)
WITH target, collect(DISTINCT {fqn: caller.fqn, repo: caller.repo,
       file_path: caller.file_path, start_line: caller.start_line,
       hops: size(c1),
       min_confidence: reduce(m = 1.0, r IN c1 | CASE WHEN r.confidence IS NULL
         THEN m ELSE (CASE WHEN r.confidence < m THEN r.confidence ELSE m END) END)
}) AS callers
OPTIONAL MATCH (target)-[c2:CALLS*1..%(depth)d]->(callee:Symbol)
WITH target, callers, collect(DISTINCT {fqn: callee.fqn, repo: callee.repo,
       file_path: callee.file_path, start_line: callee.start_line,
       hops: size(c2),
       min_confidence: reduce(m = 1.0, r IN c2 | CASE WHEN r.confidence IS NULL
         THEN m ELSE (CASE WHEN r.confidence < m THEN r.confidence ELSE m END) END)
}) AS callees
RETURN
  {fqn: target.fqn, repo: target.repo, file_path: target.file_path,
   start_line: target.start_line, end_line: target.end_line} AS target,
  [c IN callers WHERE c.fqn IS NOT NULL] AS callers,
  [c IN callees WHERE c.fqn IS NOT NULL] AS callees
"""


@audited("get_call_graph")
def get_call_graph(
    workspace_id: str,
    function_fqn: str,
    depth: int = 2,
    direction: Literal["callers", "callees", "both"] = "both",
    match_mode: MatchMode = "exact",
    max_candidates: int = 10,
) -> dict[str, Any]:
    """Return callers and callees of a Symbol up to ``depth`` hops away.

    Args:
        workspace_id:  Manifest workspace id (e.g. ``payments-platform``).
        function_fqn:  Fully-qualified name of the target symbol, or — when
            ``match_mode != 'exact'`` — a partial name / search phrase.
        depth:         1-5 hops. Higher values are exponentially more expensive.
        direction:     ``"callers"``, ``"callees"`` or ``"both"`` (default).
        match_mode:    Symbol-lookup strategy. ``"exact"`` (default) requires
            a complete FQN match — same behaviour as previous releases.
            ``"suffix"`` matches when ``s.fqn ENDS WITH function_fqn`` (tail
            of at least 3 chars). ``"fuzzy"`` runs a fulltext query over the
            ``symbol_text`` index. ``"auto"`` tries each in turn and stops
            on the first non-empty result.
        max_candidates: 1-50. Number of resolved candidates to consider.
            Only the highest-scoring candidate is used as the call-graph
            target; the rest are returned under ``candidates`` so the agent
            can disambiguate.

    Returns a dict with ``target``, ``callers``, ``callees``, ``candidates``
    and ``match_mode_used`` (``None`` when nothing was resolved). When
    ``match_mode='exact'`` and resolution succeeds, ``candidates`` has a
    single entry and ``match_mode_used == 'exact'``.
    """
    assert_safe_workspace_id(workspace_id)
    if not 1 <= depth <= 5:
        raise ValueError("depth must be between 1 and 5")
    if direction not in {"callers", "callees", "both"}:
        raise ValueError(f"invalid direction: {direction!r}")
    if match_mode not in {"exact", "suffix", "fuzzy", "auto"}:
        raise ValueError(f"invalid match_mode: {match_mode!r}")
    # Strict identifier-only validation when we will pass the input as an
    # exact FQN match (exact/suffix). Fuzzy + auto need to accept search
    # phrases (with spaces and broader punctuation) since the input is
    # Lucene-escaped and bound as a parameter.
    if match_mode in {"exact", "suffix"}:
        assert_safe_identifier(function_fqn)
    else:
        assert_safe_search_text(function_fqn)
    max_candidates = clamp_candidates(max_candidates)

    ctx = get_context()
    with read_session(ctx.neo4j) as session:
        candidates, mode_used = resolve(
            session,
            workspace_id=workspace_id,
            fqn=function_fqn,
            mode=match_mode,
            max_candidates=max_candidates,
        )
        candidate_payload = [resolved_to_payload(c) for c in candidates]
        if not candidates:
            return {
                "target": None,
                "callers": [],
                "callees": [],
                "candidates": [],
                "match_mode_used": None,
            }

        # Highest-scoring candidate drives the call-graph traversal.
        primary_fqn = candidates[0].fqn
        record = session.run(
            _QUERY % {"depth": depth},
            workspace_id=workspace_id,
            fqn=primary_fqn,
        ).single()
        if record is None:
            return {
                "target": None,
                "callers": [],
                "callees": [],
                "candidates": candidate_payload,
                "match_mode_used": mode_used,
            }

    callers = record["callers"] if direction in {"callers", "both"} else []
    callees = record["callees"] if direction in {"callees", "both"} else []
    return {
        "target": record["target"],
        "callers": callers,
        "callees": callees,
        "candidates": candidate_payload,
        "match_mode_used": mode_used,
    }
