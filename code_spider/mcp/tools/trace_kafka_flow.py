"""``trace_kafka_flow`` — producers → topic → consumers."""

from __future__ import annotations

from typing import Any

from code_spider.mcp.auth import (
    assert_safe_identifier,
    assert_safe_workspace_id,
    audited,
    read_session,
)
from code_spider.mcp.context import get_context

_QUERY = """
MATCH (t:KafkaTopic {workspace_id: $workspace_id, name: $topic})
OPTIONAL MATCH (prod:KafkaProducer)-[:WRITES_TO]->(t)
OPTIONAL MATCH (prod_caller:Symbol)-[:PRODUCES]->(prod)
OPTIONAL MATCH (cons:KafkaConsumer)-[:READS_FROM]->(t)
OPTIONAL MATCH (cons_caller:Symbol)-[:CONSUMES]->(cons)
RETURN
  t.name AS topic,
  collect(DISTINCT CASE WHEN prod IS NULL THEN NULL ELSE
    {fqn: prod_caller.fqn, repo: prod.repo, client_lib: prod.client_lib,
     file_path: prod.file_path, start_line: prod.start_line} END) AS producers,
  collect(DISTINCT CASE WHEN cons IS NULL THEN NULL ELSE
    {fqn: cons_caller.fqn, repo: cons.repo, client_lib: cons.client_lib,
     group_id: cons.group_id, file_path: cons.file_path,
     start_line: cons.start_line} END) AS consumers
"""


@audited("trace_kafka_flow")
def trace_kafka_flow(workspace_id: str, topic: str) -> dict[str, Any]:
    """Return all producers and consumers writing to / reading from ``topic``."""
    assert_safe_workspace_id(workspace_id)
    assert_safe_identifier(topic, max_len=256)

    ctx = get_context()
    with read_session(ctx.neo4j) as session:
        row = session.run(
            _QUERY, workspace_id=workspace_id, topic=topic
        ).single()
    if row is None:
        return {"topic": topic, "producers": [], "consumers": []}
    return {
        "topic": row["topic"],
        "producers": [p for p in row["producers"] if p],
        "consumers": [c for c in row["consumers"] if c],
    }
