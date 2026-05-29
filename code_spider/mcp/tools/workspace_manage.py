"""``workspace_manage`` — read-only workspace introspection."""

from __future__ import annotations

from typing import Any, Literal

from code_spider.graph.count_cache import CountEntry, get_cache
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

# Metadata-only lookup: the Workspace node plus its repo count. Cheap because
# it touches a single Workspace and its CONTAINS edges — no label scans.
_STATS_META_QUERY = """
MATCH (w:Workspace {id: $workspace_id})
OPTIONAL MATCH (w)-[:CONTAINS]->(r:Repository)
RETURN w.id AS id, w.name AS name, w.manifest_sha AS manifest_sha,
       count(DISTINCT r) AS repos
"""

# Per-label counts are run as separate, index-friendly queries rather than one
# mega-query with six OPTIONAL MATCH + count(DISTINCT) clauses. The old single
# query built a cartesian product of every Symbol × Route × Chunk × … which
# materialised millions of intermediate rows and routinely blew the tool
# timeout. Each entry maps the output key to the node label it counts; all
# carry a ``workspace_id`` property so the count is workspace-scoped.
_STATS_LABELS: tuple[tuple[str, str], ...] = (
    ("symbols", "Symbol"),
    ("routes", "Route"),
    ("http_client_calls", "HttpClientCall"),
    ("kafka_topics", "KafkaTopic"),
    ("chunks", "Chunk"),
)


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
        cache = get_cache()
        with read_session(ctx.neo4j) as session:
            meta = session.run(_STATS_META_QUERY, workspace_id=workspace_id).single()
            if meta is None or meta["id"] is None:
                return {"workspace": None, "stats": {}}

            stats: dict[str, int] = {"repos": meta["repos"]}
            # Each label count is a small index-friendly scan, memoised through
            # the shared TTL count cache so repeated stats/schema calls collapse
            # to a single round-trip per label.
            for key, label in _STATS_LABELS:

                def _fetch(_label: str = label) -> CountEntry:
                    row = session.run(
                        f"MATCH (n:`{_label}`) "
                        "WHERE n.workspace_id = $workspace_id "
                        "RETURN count(n) AS c",
                        workspace_id=workspace_id,
                    ).single()
                    return CountEntry(
                        value=int(row["c"]) if row else 0, scope="workspace"
                    )

                entry = cache.get_or_compute(
                    workspace_id=workspace_id,
                    kind="node",
                    name=label,
                    scoped=True,
                    fetch=_fetch,
                )
                stats[key] = entry.value

        return {
            "workspace": {
                "id": meta["id"],
                "name": meta["name"],
                "manifest_sha": meta["manifest_sha"],
            },
            "stats": stats,
        }
    raise ValueError(f"unknown action: {action!r}")
