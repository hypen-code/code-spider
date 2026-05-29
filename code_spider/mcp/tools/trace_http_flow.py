"""``trace_http_flow`` — full client→route→handler→downstream chain."""

from __future__ import annotations

from typing import Any

from code_spider.mcp.auth import (
    assert_safe_identifier,
    assert_safe_workspace_id,
    audited,
    read_session,
)
from code_spider.mcp.context import get_context
from code_spider.routes._common import normalize_method, normalize_path

_QUERY = """
MATCH (route:Route {workspace_id: $workspace_id})
WHERE (route.method = $method OR route.method = '*' OR $method = '*')
  AND route.path = $path
OPTIONAL MATCH (handler:Symbol)-[:HANDLES]->(route)
OPTIONAL MATCH (call:HttpClientCall)-[flow:HTTP_FLOW]->(route)
OPTIONAL MATCH (caller:Symbol)-[:INVOKES]->(call)
OPTIONAL MATCH (handler)-[:CALLS*1..3]->(downstream:Symbol)
RETURN
  {method: route.method, path: route.path, framework: route.framework,
   repo: route.repo, file_path: route.file_path,
   start_line: route.start_line} AS route,
  collect(DISTINCT CASE WHEN handler IS NULL THEN NULL ELSE
    {fqn: handler.fqn, repo: handler.repo, file_path: handler.file_path,
     start_line: handler.start_line} END) AS handlers,
  collect(DISTINCT CASE WHEN caller IS NULL THEN NULL ELSE
    {fqn: caller.fqn, repo: caller.repo, file_path: caller.file_path,
     start_line: caller.start_line, match_score: flow.match_score,
     method: call.method, path: call.path_template} END) AS callers,
  collect(DISTINCT CASE WHEN downstream IS NULL THEN NULL ELSE
    {fqn: downstream.fqn, repo: downstream.repo} END) AS downstream_symbols
"""


@audited("trace_http_flow")
def trace_http_flow(
    workspace_id: str, method: str, path: str
) -> dict[str, Any]:
    """Trace an HTTP request from client callers through the handler chain.

    Provide ``method`` (``GET``/``POST``/...) and a normalised ``path``
    (use ``{}`` for path parameters, e.g. ``/users/{}``).
    """
    assert_safe_workspace_id(workspace_id)
    method_norm = normalize_method(method)
    path_norm = normalize_path(path)
    if not method_norm:
        raise ValueError("method is required")
    assert_safe_identifier(path_norm, max_len=512)

    ctx = get_context()
    with read_session(ctx.neo4j) as session:
        rows = list(
            session.run(
                _QUERY,
                workspace_id=workspace_id,
                method=method_norm,
                path=path_norm,
            )
        )

    flows: list[dict[str, Any]] = []
    for row in rows:
        flows.append(
            {
                "route": row["route"],
                "handlers": [h for h in row["handlers"] if h],
                "callers": [c for c in row["callers"] if c],
                "downstream_symbols": [d for d in row["downstream_symbols"] if d],
            }
        )
    return {"method": method_norm, "path": path_norm, "flows": flows}
