"""MCP tool registration.

All tools are pure functions exposed via :func:`register_all`. They never
talk to MCP directly — that decoupling keeps them unit-testable and lets
us run the same logic over an HTTP API later without rewriting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_spider.mcp.tools.execute_cypher import execute_cypher
from code_spider.mcp.tools.get_call_graph import get_call_graph
from code_spider.mcp.tools.get_coordinate_snippet import get_coordinate_snippet
from code_spider.mcp.tools.get_graph_schema import get_graph_schema
from code_spider.mcp.tools.get_impact_analysis import get_impact_analysis
from code_spider.mcp.tools.index_repository import index_repository
from code_spider.mcp.tools.semantic_code_search import semantic_code_search
from code_spider.mcp.tools.trace_http_flow import trace_http_flow
from code_spider.mcp.tools.trace_kafka_flow import trace_kafka_flow
from code_spider.mcp.tools.workspace_manage import workspace_manage

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


__all__ = [
    "execute_cypher",
    "get_call_graph",
    "get_coordinate_snippet",
    "get_graph_schema",
    "get_impact_analysis",
    "index_repository",
    "register_all",
    "semantic_code_search",
    "trace_http_flow",
    "trace_kafka_flow",
    "workspace_manage",
]


def register_all(mcp: FastMCP) -> None:
    """Register every tool on the given FastMCP instance."""
    # Each call uses ``mcp.tool()`` to wrap a plain Python function. The
    # docstrings double as the tool descriptions that LLM agents see.
    mcp.tool()(get_call_graph)
    mcp.tool()(get_impact_analysis)
    mcp.tool()(semantic_code_search)
    mcp.tool()(get_coordinate_snippet)
    mcp.tool()(trace_http_flow)
    mcp.tool()(trace_kafka_flow)
    mcp.tool()(workspace_manage)
    mcp.tool()(index_repository)
    # ``get_graph_schema`` is registered just before ``execute_cypher`` so
    # tools that introspect listings (FastMCP, Claude, etc.) surface the
    # schema tool right next to the ad-hoc query tool. Both docstrings
    # carry a STRICT INVOCATION RULE: the schema is only fetched when
    # the agent is about to use execute_cypher, and the response is
    # cached for the rest of the session.
    mcp.tool()(get_graph_schema)
    mcp.tool()(execute_cypher)
