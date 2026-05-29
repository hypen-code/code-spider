"""``execute_cypher`` MCP tool — read-only ad-hoc Cypher access.

The fixed-shape tools (``get_call_graph``, ``semantic_code_search``, etc.)
cover the common code-navigation queries, but agents sometimes need to
ask the graph something custom. This tool lets them issue raw Cypher
**provided it is strictly read-only**.

Read-only enforcement is layered defence-in-depth:

1. **Static keyword scan** (cheap pre-flight). Strings, backticked
   identifiers and comments are stripped before scanning so query text
   like ``WHERE name = 'CREATE foo'`` does not false-trip.
2. **``EXPLAIN`` query-type check** (authoritative). Neo4j plans the
   statement without executing it and classifies it as ``"r"`` /
   ``"rw"`` / ``"w"`` / ``"s"``. Anything except ``"r"`` is rejected.
3. **READ-mode session** + managed read transaction with a hard
   ``timeout`` so a runaway query cannot pin a worker.

Single-statement only, length capped, results capped, parameters
validated for JSON compatibility, output rows JSON-serialised so
Nodes/Relationships/Paths cross the MCP boundary cleanly.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from neo4j.graph import Node, Path, Relationship

from code_spider.mcp.auth import (
    assert_safe_workspace_id,
    audited,
    read_session,
)
from code_spider.mcp.context import get_context

# ---------- Bounds / config ------------------------------------------------ #

MAX_QUERY_CHARS: int = 10_000
MAX_LIMIT: int = 1000
DEFAULT_LIMIT: int = 100
TRANSACTION_TIMEOUT_SECONDS: float = 30.0

# ---------- Regexes used to strip non-code spans before scanning ----------- #

_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", re.DOTALL)
_BACKTICK_IDENT = re.compile(r"`(?:[^`\\]|\\.)*`")
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# Whole-word write keywords. Order does not matter; we just need ANY match
# to short-circuit before sending the statement to Neo4j.
_WRITE_KEYWORDS = re.compile(
    r"\b(?:"
    r"CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|"
    r"LOAD\s+CSV|USING\s+PERIODIC\s+COMMIT|"
    r"GRANT|REVOKE|DENY|ALTER|RENAME|"
    r"START\s+DATABASE|STOP\s+DATABASE"
    r")\b",
    re.IGNORECASE,
)


# ---------- Pre-flight static checks --------------------------------------- #


def _scrub(query: str) -> str:
    """Remove string literals, backticked identifiers and comments so the
    keyword scan sees only Cypher *code*."""
    s = _BLOCK_COMMENT.sub(" ", query)
    s = _LINE_COMMENT.sub(" ", s)
    s = _STRING_LITERAL.sub("''", s)
    s = _BACKTICK_IDENT.sub("`x`", s)
    return s


def _check_no_writes(query: str) -> None:
    """Raise ``PermissionError`` if a write keyword appears at code level."""
    scrubbed = _scrub(query)
    m = _WRITE_KEYWORDS.search(scrubbed)
    if m:
        keyword = re.sub(r"\s+", " ", m.group(0)).upper()
        raise PermissionError(
            f"write keyword '{keyword}' is not allowed; "
            "execute_cypher is read-only"
        )


def _check_single_statement(query: str) -> None:
    """Reject queries with internal ``;`` separators (multi-statement)."""
    scrubbed = _scrub(query).rstrip().rstrip(";").rstrip()
    if ";" in scrubbed:
        raise ValueError("only a single Cypher statement is permitted")


def _validate_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure ``parameters`` is a flat JSON-compatible mapping."""
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a JSON object (dict)")
    allowed: tuple[type, ...] = (str, int, float, bool, list, dict, type(None))
    for key, value in parameters.items():
        if not isinstance(key, str):
            raise ValueError(f"parameter key {key!r} must be a string")
        if not isinstance(value, allowed):
            raise ValueError(
                f"parameter {key!r} has unsupported type {type(value).__name__}; "
                "only JSON primitives, lists and dicts are allowed"
            )
    return parameters


# ---------- Graph-type → JSON serialisation -------------------------------- #


def _serialize_value(value: Any) -> Any:
    """Recursively coerce Neo4j graph types to JSON-friendly primitives."""
    if isinstance(value, Node):
        return {
            "_type": "node",
            "element_id": value.element_id,
            "labels": sorted(value.labels),
            "properties": dict(value),
        }
    if isinstance(value, Relationship):
        start_id = value.start_node.element_id if value.start_node else None
        end_id = value.end_node.element_id if value.end_node else None
        return {
            "_type": "relationship",
            "element_id": value.element_id,
            "type": value.type,
            "start": start_id,
            "end": end_id,
            "properties": dict(value),
        }
    if isinstance(value, Path):
        return {
            "_type": "path",
            "nodes": [_serialize_value(n) for n in value.nodes],
            "relationships": [_serialize_value(r) for r in value.relationships],
        }
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize_value(x) for x in value]
    # Neo4j temporal types subclass the std-lib ones; ISO 8601 is the most
    # portable representation across JSON / agents / clients.
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return value


# ---------- The MCP tool --------------------------------------------------- #


@audited("execute_cypher")
def execute_cypher(
    workspace_id: str,
    query: str,
    parameters: dict[str, Any] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Execute a **read-only** Cypher statement and return the raw rows.

    Use this when the fixed-shape tools (``get_call_graph``,
    ``semantic_code_search``, ``trace_http_flow`` …) cannot express what
    you need. WRITES ARE STRICTLY REJECTED — any statement whose Neo4j
    ``EXPLAIN`` query-type is not ``"r"`` raises ``PermissionError``.

    **Before your FIRST call to this tool**: invoke ``get_graph_schema``
    once and cache its response. It is the only way to discover the
    actual labels, relationship types, properties, and index names — the
    graph also follows a strict ``workspace_id`` scoping convention which
    the schema response documents. Without that context an LLM will
    hallucinate property names and waste round-trips on
    ``invalid Cypher`` errors.

    ``get_graph_schema`` is expensive (~300 ms, ~30 KB), so only call it
    **once per session**, only when you are about to use this tool, and
    reuse the cached response for every subsequent ``execute_cypher``
    call. If you only need fixed-shape tools (``get_call_graph``,
    ``semantic_code_search``, ``trace_*``, etc.), skip the schema fetch
    entirely.

    Always scope the statement to a workspace yourself, e.g.::

        MATCH (s:Symbol {workspace_id: $workspace_id})
        WHERE s.kind = 'function'
        RETURN s.fqn, s.repo, s.file_path
        LIMIT 25

    Args:
        workspace_id: Manifest workspace id. Used for the audit log; the
            statement still has to scope its own filters (e.g. with a
            ``$workspace_id`` parameter that you pass in ``parameters``).
        query: A single read-only Cypher statement (max 10 000 chars).
            Multi-statement queries are rejected.
        parameters: Optional dict of Cypher parameters (JSON-compatible
            values only). Reference them with ``$name`` in the query.
        limit: Maximum rows to return (1-1000, default 100). The response
            sets ``truncated=true`` if more rows were available.

    Returns:
        Mapping with::

            {
              "columns": ["fqn", "repo", ...],
              "rows": [{"fqn": "...", ...}, ...],
              "row_count": N,
              "truncated": bool,
              "query_type": "r",
              "result_available_after_ms": int,
              "result_consumed_after_ms": int,
            }

        Nodes/Relationships/Paths are serialised as dicts carrying
        ``_type``, ``element_id``, ``labels``/``type`` and ``properties``.

    Raises:
        ValueError: malformed query, bad parameters, or out-of-range limit.
        PermissionError: the statement is not read-only.
    """
    assert_safe_workspace_id(workspace_id)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query exceeds {MAX_QUERY_CHARS} characters")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")

    _check_single_statement(query)
    _check_no_writes(query)
    params = _validate_parameters(parameters)

    ctx = get_context()
    with read_session(ctx.neo4j) as session:
        # Layer 2 — authoritative read-only gate.
        try:
            explain_summary = session.run(
                f"EXPLAIN {query}", parameters=params
            ).consume()
        except Exception as exc:  # pragma: no cover - exercised in live runs
            raise ValueError(f"invalid Cypher: {exc}") from exc
        query_type = explain_summary.query_type
        if query_type != "r":
            raise PermissionError(
                f"only read-only queries are allowed "
                f"(query_type={query_type!r}); execute_cypher rejected the request"
            )

        # Layer 3 — explicit read transaction with a hard timeout. We use
        # ``begin_transaction`` rather than ``execute_read`` because the
        # latter forwards extra kwargs to the work function and offers no
        # transaction-level timeout knob. For a read-only ad-hoc query we
        # also do not need the auto-retry that ``execute_read`` provides.
        with session.begin_transaction(timeout=TRANSACTION_TIMEOUT_SECONDS) as tx:
            payload = _run_and_collect(
                tx,
                query=query,
                params=params,
                limit=limit,
                query_type=query_type,
            )
            # Explicit commit (no-op for a read but suppresses any
            # "transaction not committed" driver warnings).
            tx.commit()
        return payload


def _run_and_collect(
    tx: Any,
    *,
    query: str,
    params: dict[str, Any],
    limit: int,
    query_type: str,
) -> dict[str, Any]:
    """Transaction body: run, page rows up to ``limit``, JSON-serialise."""
    result = tx.run(query, parameters=params)
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    truncated = False
    for record in result:
        if not columns:
            columns = list(record.keys())
        if len(rows) >= limit:
            truncated = True
            break
        rows.append({k: _serialize_value(record[k]) for k in columns})
    # Drain the rest so the summary timings are final.
    if truncated:
        for _ in result:
            pass
    summary = result.consume()
    if not columns:
        # Empty result: try to recover column names from the driver.
        try:
            columns = list(result.keys())
        except Exception:  # pragma: no cover - driver-specific fallback
            columns = []
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "query_type": query_type,
        "result_available_after_ms": summary.result_available_after,
        "result_consumed_after_ms": summary.result_consumed_after,
    }
