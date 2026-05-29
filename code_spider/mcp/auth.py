"""Read-only safety net for MCP tools.

The MCP server never exposes raw Cypher to agents — every tool runs a
fixed, parameterised query. As defence-in-depth we additionally:

    1. Use Neo4j **read sessions** for every tool query (``default_access_mode='READ'``).
    2. Validate every tool's parameters against a strict allow-list of values
       (workspace IDs must come from the manifest; identifiers must match a
       safe regex).
    3. Emit an audit log entry per invocation via structlog.

Phase 2 hardening — kept intentionally small so future RBAC integration
(Enterprise upgrade) is a drop-in replacement.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from neo4j import READ_ACCESS, Session

from code_spider.graph.client import Neo4jClient
from code_spider.logging_setup import get_logger
from code_spider.observability import METRICS

_log = get_logger("code_spider.mcp.audit")


# Identifiers we are willing to interpolate into MERGE/MATCH parameters.
_SAFE_IDENT = re.compile(r"^[A-Za-z0-9_\.\-:/@*\{\}]+$")
_SAFE_WORKSPACE_ID = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


def assert_safe_workspace_id(value: str) -> str:
    if not _SAFE_WORKSPACE_ID.fullmatch(value or ""):
        raise ValueError(f"invalid workspace_id: {value!r}")
    return value


def assert_safe_identifier(value: str, *, max_len: int = 512) -> str:
    if not value or len(value) > max_len or not _SAFE_IDENT.fullmatch(value):
        raise ValueError(f"invalid identifier value: {value!r}")
    return value


def read_session(client: Neo4jClient) -> Session:
    """Return a session pinned to READ access mode."""
    # Reach through the wrapper to the underlying driver to set access mode.
    driver = client._driver
    return driver.session(
        database=client._settings.database,
        default_access_mode=READ_ACCESS,
    )


T = TypeVar("T")


def audited(tool_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: audit-log every tool invocation and record Prometheus timings."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            started = time.perf_counter()
            audit_kwargs = {k: v for k, v in kwargs.items() if k != "query_text"}
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                duration = time.perf_counter() - started
                METRICS.mcp_tool_duration.labels(tool=tool_name).observe(duration)
                METRICS.mcp_tool_errors.labels(tool=tool_name).inc()
                _log.warning(
                    "tool.error",
                    tool=tool_name,
                    error=str(exc),
                    duration_ms=round(duration * 1000, 1),
                    **audit_kwargs,
                )
                raise
            duration = time.perf_counter() - started
            METRICS.mcp_tool_duration.labels(tool=tool_name).observe(duration)
            _log.info(
                "tool.ok",
                tool=tool_name,
                duration_ms=round(duration * 1000, 1),
                **audit_kwargs,
            )
            return result

        return wrapper

    return decorator
