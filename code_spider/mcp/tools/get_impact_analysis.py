"""``get_impact_analysis`` — downstream symbols + cross-service reach."""

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
MATCH (s:Symbol {workspace_id: $workspace_id, fqn: $fqn})

// Downstream callers via transitive :CALLS (up to 5 hops).
OPTIONAL MATCH (caller:Symbol)-[:CALLS*1..5]->(s)
WITH s, collect(DISTINCT {fqn: caller.fqn, repo: caller.repo,
   file_path: caller.file_path, start_line: caller.start_line}) AS callers

// Routes whose handler is impacted (s is the handler or transitively calls one).
OPTIONAL MATCH path = (s)-[:CALLS*0..3]->(handler:Symbol)-[:HANDLES]->(route:Route)
WITH s, callers, collect(DISTINCT {method: route.method, path: route.path,
   framework: route.framework, repo: route.repo,
   handler_fqn: handler.fqn, file_path: route.file_path}) AS handles_routes

// HTTP_FLOW reach: external client calls that ultimately invoke this symbol.
OPTIONAL MATCH (client_caller:Symbol)-[:INVOKES]->(call:HttpClientCall)
              -[:HTTP_FLOW]->(:Route)<-[:HANDLES]-(:Symbol)-[:CALLS*0..5]->(s)
WITH s, callers, handles_routes,
     collect(DISTINCT {fqn: client_caller.fqn, repo: client_caller.repo,
       method: call.method, path: call.path_template,
       file_path: client_caller.file_path}) AS http_flow_inbound

// Kafka producers reaching this symbol via published topics.
OPTIONAL MATCH (prod:KafkaProducer)<-[:PRODUCES]-(:Symbol)-[:CALLS*0..5]->(s)
WITH s, callers, handles_routes, http_flow_inbound,
     collect(DISTINCT {topic: prod.topic_name, repo: prod.repo,
       file_path: prod.file_path}) AS kafka_inbound

RETURN
  {fqn: s.fqn, repo: s.repo, file_path: s.file_path,
   start_line: s.start_line, end_line: s.end_line} AS target,
  [x IN callers WHERE x.fqn IS NOT NULL] AS upstream_symbols,
  [x IN handles_routes WHERE x.path IS NOT NULL] AS handles_routes,
  [x IN http_flow_inbound WHERE x.fqn IS NOT NULL] AS http_inbound,
  [x IN kafka_inbound WHERE x.topic IS NOT NULL] AS kafka_inbound
"""


@audited("get_impact_analysis")
def get_impact_analysis(workspace_id: str, symbol_fqn: str) -> dict[str, Any]:
    """Surface everything that depends on ``symbol_fqn``.

    Returns upstream callers, REST routes whose handler chain touches the
    symbol, and any cross-service HTTP_FLOW + Kafka producers that reach it.

    Use this before refactoring a symbol to estimate blast radius.
    """
    assert_safe_workspace_id(workspace_id)
    assert_safe_identifier(symbol_fqn)

    ctx = get_context()
    with read_session(ctx.neo4j) as session:
        record = session.run(
            _QUERY, workspace_id=workspace_id, fqn=symbol_fqn
        ).single()
    if record is None:
        return {
            "target": None,
            "upstream_symbols": [],
            "handles_routes": [],
            "http_inbound": [],
            "kafka_inbound": [],
        }
    return {
        "target": record["target"],
        "upstream_symbols": record["upstream_symbols"],
        "handles_routes": record["handles_routes"],
        "http_inbound": record["http_inbound"],
        "kafka_inbound": record["kafka_inbound"],
    }
