"""Code Spider MCP server — `FastMCP` instance with the 8-tool surface."""

from __future__ import annotations

from typing import Final

from mcp.server.fastmcp import FastMCP

from code_spider.config import Settings, load_settings
from code_spider.mcp.context import initialize, shutdown
from code_spider.mcp.tools import register_all

SERVER_NAME: Final = "code-spider"
SERVER_DESCRIPTION: Final = (
    "Centralized codebase knowledge graph. Tools expose coordinate-aware "
    "navigation (call graphs, impact analysis, REST/Kafka flow tracing) and "
    "hybrid lexical+vector code search."
)


def build_server() -> FastMCP:
    """Construct the MCP server and register every tool. Pure; no side effects
    until :meth:`FastMCP.run` is called."""
    mcp = FastMCP(SERVER_NAME, instructions=SERVER_DESCRIPTION)
    register_all(mcp)
    return mcp


def run_stdio(
    *,
    settings: Settings | None = None,
    embed_provider: str = "auto",
) -> None:
    """Bring up the MCP server on the stdio transport (the MCP default)."""
    initialize(settings=settings or load_settings(), embed_provider=embed_provider)
    try:
        server = build_server()
        server.run("stdio")
    finally:
        shutdown()
