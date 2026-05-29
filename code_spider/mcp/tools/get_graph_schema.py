"""``get_graph_schema`` MCP tool — full read-only schema introspection.

Returns labels, relationship endpoints, properties, indexes and
constraints in one payload. Read-only: ``db.schema.*`` + ``SHOW INDEXES``
/ ``SHOW CONSTRAINTS`` over a READ session.

Expensive (~300 ms, 20-40 KB). Call **only before ``execute_cypher``**,
cache the response for the session. Skip entirely if you are only using
the fixed-shape tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from code_spider.mcp.auth import (
    assert_safe_workspace_id,
    audited,
    read_session,
)
from code_spider.mcp.context import get_context


@dataclass(frozen=True, slots=True)
class _Count:
    """Internal struct paired with each label/rel-type during aggregation."""

    value: int
    scope: Literal["workspace", "global"]

# Identifiers we are willing to interpolate into ad-hoc count queries.
# Labels and relationship types come from Neo4j itself, but we still
# require them to match a strict regex before formatting them into Cypher
# to make the code obviously injection-proof.
_SAFE_LABEL_OR_REL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Cypher fragments. Kept as module-level constants so they show up in
# tracebacks and can be reused by tests.
_NODE_TYPE_PROPS = "CALL db.schema.nodeTypeProperties()"
_REL_TYPE_PROPS = "CALL db.schema.relTypeProperties()"
_VISUALIZATION = "CALL db.schema.visualization()"
_SHOW_INDEXES = (
    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, "
    "properties, state, options"
)
_SHOW_CONSTRAINTS = (
    "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties"
)


# --------------------------------------------------------------------------- #
# Pure aggregators (unit-tested without Neo4j)                                #
# --------------------------------------------------------------------------- #


def _aggregate_node_props(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Fold ``CALL db.schema.nodeTypeProperties`` rows into a label→props map."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        labels = row.get("nodeLabels") or []
        if not labels:
            continue
        # Most labels are single; multi-labeled types repeat the same prop
        # under each label, which is fine — we want them all listed there.
        prop_name = row.get("propertyName")
        for label in labels:
            bucket = out.setdefault(label, [])
            if prop_name is None:
                continue
            bucket.append(
                {
                    "name": prop_name,
                    "types": list(row.get("propertyTypes") or []),
                    "mandatory": bool(row.get("mandatory", False)),
                }
            )
    # Multi-labelled nodes (e.g. ``:Symbol:Function``) cause the same
    # property to appear in several rows under the same label. Merge by
    # property name, unioning the type set and OR-ing ``mandatory``.
    for label in list(out.keys()):
        out[label] = _dedupe_props(out[label])
    return out


def _aggregate_rel_props(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Fold ``CALL db.schema.relTypeProperties`` rows into a relType→props map."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rel_type = _strip_quoted_type(row.get("relType"))
        if not rel_type:
            continue
        bucket = out.setdefault(rel_type, [])
        prop_name = row.get("propertyName")
        if prop_name is None:
            continue
        bucket.append(
            {
                "name": prop_name,
                "types": list(row.get("propertyTypes") or []),
                "mandatory": bool(row.get("mandatory", False)),
            }
        )
    for rt in list(out.keys()):
        out[rt] = _dedupe_props(out[rt])
    return out


def _dedupe_props(
    props: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse duplicate property descriptors by name (case-sensitive).

    ``types`` is unioned across duplicates; ``mandatory`` is OR-ed (a
    property is mandatory iff it is mandatory in *every* label combo —
    we approximate as "mandatory in any" so the LLM never assumes
    a value will be present when it might not be). Output is sorted by
    name for deterministic rendering.
    """
    merged: dict[str, dict[str, Any]] = {}
    for p in props:
        name = p["name"]
        cur = merged.get(name)
        if cur is None:
            merged[name] = {
                "name": name,
                "types": sorted(set(p.get("types") or [])),
                "mandatory": bool(p.get("mandatory", False)),
            }
            continue
        cur["types"] = sorted(set(cur["types"]) | set(p.get("types") or []))
        cur["mandatory"] = bool(cur["mandatory"]) and bool(p.get("mandatory", False))
    return sorted(merged.values(), key=lambda x: x["name"])


def _strip_quoted_type(value: Any) -> str | None:
    """``db.schema.relTypeProperties`` returns ``relType`` as ``":`CALLS`"``."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip(":` ")
    return cleaned or None


def _aggregate_endpoints(
    nodes: list[Any],
    relationships: list[Any],
) -> dict[str, list[dict[str, str]]]:
    """Walk ``db.schema.visualization`` output into relType → [{start,end}…]."""
    label_by_id: dict[str, str] = {}
    for node in nodes:
        try:
            # virtual nodes carry their label name in the ``name`` property
            label = node["name"] if "name" in list(node.keys()) else None
        except Exception:
            label = None
        if label:
            label_by_id[node.element_id] = label

    out: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for rel in relationships:
        rt = getattr(rel, "type", None)
        start = (
            label_by_id.get(rel.start_node.element_id) if rel.start_node else None
        )
        end = label_by_id.get(rel.end_node.element_id) if rel.end_node else None
        if not (rt and start and end):
            continue
        key = (rt, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(rt, []).append({"start": start, "end": end})
    for rt, eps in out.items():
        out[rt] = sorted(eps, key=lambda e: (e["start"], e["end"]))
    return out


# --------------------------------------------------------------------------- #
# Count helpers (do touch Neo4j)                                              #
# --------------------------------------------------------------------------- #


def _node_count_query(label: str, *, scoped: bool) -> str:
    """Cypher to count nodes of ``label``.

    ``scoped=True`` filters on ``n.workspace_id = $workspace_id``. Use
    ``False`` for labels that do not carry a ``workspace_id`` property
    (e.g. ``Workspace`` itself), where the global count is the right
    semantic.
    """
    if not _SAFE_LABEL_OR_REL.fullmatch(label):
        raise ValueError(f"unsafe label name: {label!r}")
    if scoped:
        return (
            f"MATCH (n:`{label}`) "
            "WHERE n.workspace_id = $workspace_id "
            "RETURN count(n) AS c"
        )
    return f"MATCH (n:`{label}`) RETURN count(n) AS c"


def _rel_count_query(rel_type: str, *, scoped: bool) -> str:
    """Cypher to count relationships of ``rel_type``.

    Relationships in this graph do not all carry ``workspace_id``; when
    ``scoped`` is True we filter on the start node's ``workspace_id``.
    """
    if not _SAFE_LABEL_OR_REL.fullmatch(rel_type):
        raise ValueError(f"unsafe relationship type: {rel_type!r}")
    if scoped:
        return (
            f"MATCH (a)-[r:`{rel_type}`]->() "
            "WHERE a.workspace_id = $workspace_id "
            "RETURN count(r) AS c"
        )
    return f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS c"


# --------------------------------------------------------------------------- #
# Static guidance — short prose the LLM gets every call                       #
# --------------------------------------------------------------------------- #


_GUIDANCE: tuple[str, ...] = (
    "Every primary node (Workspace, Repository, File, Module, Symbol, "
    "Chunk, Route, HttpClientCall, KafkaTopic, KafkaProducer, KafkaConsumer) "
    "carries a 'workspace_id' string property. Always scope ad-hoc queries "
    "with `WHERE n.workspace_id = $workspace_id` to avoid cross-tenant bleed.",
    "Pass `workspace_id` to execute_cypher via the `parameters` dict, e.g. "
    "`parameters={'workspace_id': 'demo'}`, then reference it as `$workspace_id`.",
    "Fulltext index 'symbol_text' covers (:Symbol) on [name, signature, "
    "docstring]. Query it with `CALL db.index.fulltext.queryNodes("
    "'symbol_text', $q) YIELD node, score`.",
    "HNSW vector index 'chunk_embedding' covers (:Chunk).embedding with "
    "cosine similarity at 384 dimensions. Use semantic_code_search rather "
    "than calling the index directly so the embed provider is consistent.",
    "Relationship endpoints in `relationship_types[*].endpoints` show the "
    "(start_label, end_label) pairs that actually occur — use these to "
    "compose `MATCH (a:X)-[r:REL]->(b:Y)` correctly.",
)


# --------------------------------------------------------------------------- #
# The MCP tool                                                                #
# --------------------------------------------------------------------------- #


@audited("get_graph_schema")
def get_graph_schema(
    workspace_id: str | None = None,
    include_counts: bool = True,
) -> dict[str, Any]:
    """Return the Neo4j schema (labels, rels, properties, indexes, constraints).

    ⚠️ Expensive (~300 ms, 20-40 KB). Call ONLY when about to use
    ``execute_cypher`` and no cached schema exists for this session;
    cache the response and reuse it. Skip entirely for fixed-shape tools.

    Args:
        workspace_id: Scope counts to this workspace (counts are global
            when ``None``). Rels are scoped via the start node.
        include_counts: Pass ``False`` to skip per-label/per-rel counts
            (~7x faster, ~half the payload).

    Returns:
        ``{node_labels, relationship_types, indexes, constraints,
        workspace_id, counts_scope, guidance}``. Each ``count`` is
        ``{value, scope: "workspace"|"global"}`` or ``None``.
    """
    if workspace_id is not None:
        assert_safe_workspace_id(workspace_id)

    ctx = get_context()
    with read_session(ctx.neo4j) as session:
        # 1. Property catalogues.
        node_prop_rows = [dict(r) for r in session.run(_NODE_TYPE_PROPS)]
        rel_prop_rows = [dict(r) for r in session.run(_REL_TYPE_PROPS)]
        node_props = _aggregate_node_props(node_prop_rows)
        rel_props = _aggregate_rel_props(rel_prop_rows)

        # 2. Endpoint structure from the visualization procedure.
        viz = session.run(_VISUALIZATION).single()
        endpoints = (
            _aggregate_endpoints(viz["nodes"], viz["relationships"])
            if viz is not None
            else {}
        )

        # 3. Indexes + constraints.
        indexes = [_index_row(r) for r in session.run(_SHOW_INDEXES)]
        constraints = [_constraint_row(r) for r in session.run(_SHOW_CONSTRAINTS)]

        # 4. Counts (optional). Per-label scope: if a label/rel type does
        # not expose ``workspace_id`` (e.g. the ``Workspace`` node itself),
        # we fall back to a global count rather than returning ``0``.
        node_counts: dict[str, _Count] = {}
        rel_counts: dict[str, _Count] = {}
        if include_counts:
            for label, props in node_props.items():
                if not _SAFE_LABEL_OR_REL.fullmatch(label):
                    continue
                has_ws = any(p["name"] == "workspace_id" for p in props)
                scoped = bool(workspace_id) and has_ws
                row = session.run(
                    _node_count_query(label, scoped=scoped),
                    parameters=({"workspace_id": workspace_id} if scoped else {}),
                ).single()
                node_counts[label] = _Count(
                    value=int(row["c"]) if row else 0,
                    scope="workspace" if scoped else "global",
                )
            for rt in rel_props:
                if not _SAFE_LABEL_OR_REL.fullmatch(rt):
                    continue
                # We can always scope on the start node's workspace_id when
                # the caller supplied one; rel-type property catalogue is
                # not consulted because the scope lives on the *node*.
                scoped = bool(workspace_id)
                row = session.run(
                    _rel_count_query(rt, scoped=scoped),
                    parameters=({"workspace_id": workspace_id} if scoped else {}),
                ).single()
                rel_counts[rt] = _Count(
                    value=int(row["c"]) if row else 0,
                    scope="workspace" if scoped else "global",
                )

    # 5. Compose the response.
    def _node_count_payload(label: str) -> dict[str, Any] | None:
        if not include_counts:
            return None
        c = node_counts.get(label)
        return None if c is None else {"value": c.value, "scope": c.scope}

    def _rel_count_payload(rt: str) -> dict[str, Any] | None:
        if not include_counts:
            return None
        c = rel_counts.get(rt)
        return None if c is None else {"value": c.value, "scope": c.scope}

    node_labels = [
        {
            "label": label,
            "count": _node_count_payload(label),
            "properties": props,
        }
        for label, props in sorted(node_props.items())
    ]
    # Union the rel types from both the property catalogue and the
    # endpoint catalogue so we never miss one.
    all_rel_types = sorted(set(rel_props) | set(endpoints))
    relationship_types = [
        {
            "type": rt,
            "count": _rel_count_payload(rt),
            "endpoints": endpoints.get(rt, []),
            "properties": rel_props.get(rt, []),
        }
        for rt in all_rel_types
    ]

    counts_scope = (
        "disabled"
        if not include_counts
        else ("workspace" if workspace_id else "global")
    )

    return {
        "node_labels": node_labels,
        "relationship_types": relationship_types,
        "indexes": indexes,
        "constraints": constraints,
        "workspace_id": workspace_id,
        "counts_scope": counts_scope,
        "guidance": list(_GUIDANCE),
    }


def _index_row(record: Any) -> dict[str, Any]:
    return {
        "name": record["name"],
        "type": record["type"],
        "entity_type": record["entityType"],
        "labels_or_types": list(record["labelsOrTypes"] or []),
        "properties": list(record["properties"] or []),
        "state": record["state"],
        "options": dict(record["options"] or {}),
    }


def _constraint_row(record: Any) -> dict[str, Any]:
    return {
        "name": record["name"],
        "type": record["type"],
        "entity_type": record["entityType"],
        "labels_or_types": list(record["labelsOrTypes"] or []),
        "properties": list(record["properties"] or []),
    }
