"""``workspace_manage`` — read-only workspace introspection."""

from __future__ import annotations

from typing import Any, Literal

from code_spider.mcp.auth import (
    assert_safe_workspace_id,
    audited,
    read_session,
)
from code_spider.mcp.context import get_context

_LIST_QUERY = """
MATCH (w:Workspace)
OPTIONAL MATCH (w)-[:CONTAINS]->(r:Repository)
RETURN w.id AS id, w.name AS name, w.manifest_sha AS manifest_sha,
       collect(DISTINCT r.name) AS repos
ORDER BY w.id
"""

_STATS_QUERY = """
MATCH (w:Workspace {id: $workspace_id})
OPTIONAL MATCH (w)-[:CONTAINS]->(r:Repository)
OPTIONAL MATCH (s:Symbol {workspace_id: $workspace_id})
OPTIONAL MATCH (rt:Route {workspace_id: $workspace_id})
OPTIONAL MATCH (h:HttpClientCall {workspace_id: $workspace_id})
OPTIONAL MATCH (ckt:KafkaTopic {workspace_id: $workspace_id})
OPTIONAL MATCH (ck:Chunk {workspace_id: $workspace_id})
RETURN
  w.id AS id, w.name AS name, w.manifest_sha AS manifest_sha,
  count(DISTINCT r) AS repos,
  count(DISTINCT s) AS symbols,
  count(DISTINCT rt) AS routes,
  count(DISTINCT h) AS http_client_calls,
  count(DISTINCT ckt) AS kafka_topics,
  count(DISTINCT ck) AS chunks
"""


@audited("workspace_manage")
def workspace_manage(
    action: Literal["list", "stats"],
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Inspect indexed workspaces.

    Actions:
        - ``list``  — every workspace known to the graph.
        - ``stats`` — node/edge counts for one workspace (``workspace_id`` required).

    All operations are read-only.
    """
    ctx = get_context()
    if action == "list":
        with read_session(ctx.neo4j) as session:
            rows = list(session.run(_LIST_QUERY))
        return {
            "workspaces": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "manifest_sha": r["manifest_sha"],
                    "repos": r["repos"],
                }
                for r in rows
            ]
        }
    if action == "stats":
        if not workspace_id:
            raise ValueError("stats requires workspace_id")
        assert_safe_workspace_id(workspace_id)
        with read_session(ctx.neo4j) as session:
            row = session.run(_STATS_QUERY, workspace_id=workspace_id).single()
        if row is None or row["id"] is None:
            return {"workspace": None, "stats": {}}
        return {
            "workspace": {
                "id": row["id"],
                "name": row["name"],
                "manifest_sha": row["manifest_sha"],
            },
            "stats": {
                "repos": row["repos"],
                "symbols": row["symbols"],
                "routes": row["routes"],
                "http_client_calls": row["http_client_calls"],
                "kafka_topics": row["kafka_topics"],
                "chunks": row["chunks"],
            },
        }
    raise ValueError(f"unknown action: {action!r}")
