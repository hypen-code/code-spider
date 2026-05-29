"""MCP server exposing the 8-tool agent surface.

Tools:
    index_repository, get_call_graph, get_impact_analysis,
    semantic_code_search, get_coordinate_snippet,
    trace_http_flow, trace_kafka_flow, workspace_manage.
"""

from code_spider.mcp.server import build_server, run_stdio

__all__ = ["build_server", "run_stdio"]
