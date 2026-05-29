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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps
from typing import Any, TypeVar

from neo4j import READ_ACCESS, Session

from code_spider.config import _DEFAULT_INDEX_TIMEOUT_S, _DEFAULT_TOOL_TIMEOUT_S
from code_spider.graph.client import Neo4jClient
from code_spider.logging_setup import get_logger
from code_spider.observability import METRICS

_log = get_logger("code_spider.mcp.audit")

#: Shared executor used to enforce a per-call wall-clock timeout on tools.
#: FastMCP already dispatches sync tools on a worker thread; running the tool
#: body on a second daemon thread lets us return a ``TimeoutError`` to the
#: agent promptly even when the underlying query is still blocked. The
#: orphaned worker drains in the background (the Neo4j session it owns is
#: closed when its ``with`` block unwinds), so we never wait on it.
_TOOL_EXECUTOR = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="codespider-tool"
)

#: Static fallback per ``Settings`` attribute, used when no live
#: :class:`ServerContext` exists (e.g. tools invoked directly from tests).
_TIMEOUT_FALLBACKS: dict[str, float] = {
    "tool_timeout_s": _DEFAULT_TOOL_TIMEOUT_S,
    "index_timeout_s": _DEFAULT_INDEX_TIMEOUT_S,
}


def _resolve_tool_timeout_s(setting_attr: str = "tool_timeout_s") -> float:
    """Return the configured timeout (seconds) for ``setting_attr``.

    Reads the named field off ``settings`` on the live
    :class:`ServerContext` when one exists (the normal server path). Falls
    back to the static default when the context is not initialised — e.g.
    tools invoked directly from unit tests — so the decorator never hard-fails.
    """
    try:
        from code_spider.mcp.context import get_context

        return float(getattr(get_context().settings, setting_attr))
    except Exception:  # noqa: BLE001 — context optional outside the server
        return _TIMEOUT_FALLBACKS.get(setting_attr, _DEFAULT_TOOL_TIMEOUT_S)


# Identifiers we are willing to interpolate into MERGE/MATCH parameters.
_SAFE_IDENT = re.compile(r"^[A-Za-z0-9_\.\-:/@*\{\}]+$")
_SAFE_WORKSPACE_ID = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
# Search text (fuzzy / fulltext) is bound as a Cypher parameter (never
# interpolated) and Lucene-escaped before being sent to the fulltext
# procedure, so we can safely accept whitespace and a wider character
# class than ``_SAFE_IDENT``. Control characters are still rejected.
_SAFE_SEARCH_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")


def assert_safe_workspace_id(value: str) -> str:
    if not _SAFE_WORKSPACE_ID.fullmatch(value or ""):
        raise ValueError(f"invalid workspace_id: {value!r}")
    return value


def assert_safe_identifier(value: str, *, max_len: int = 512) -> str:
    if not value or len(value) > max_len or not _SAFE_IDENT.fullmatch(value):
        raise ValueError(f"invalid identifier value: {value!r}")
    return value


def assert_safe_search_text(value: str, *, max_len: int = 256) -> str:
    """Validate a free-form search phrase for fuzzy/fulltext queries.

    Permits whitespace and most printable characters, but rejects empty
    strings, control characters and anything over ``max_len``. The result
    is always passed through Lucene escaping before reaching Neo4j, so the
    only attack surface is denial-of-service via gigantic inputs — which
    the length cap addresses.
    """
    if not value or len(value) > max_len or not _SAFE_SEARCH_TEXT.fullmatch(value):
        raise ValueError(f"invalid search text: {value!r}")
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


def audited(
    tool_name: str, *, timeout_setting: str = "tool_timeout_s"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: audit-log every tool invocation and record Prometheus timings.

    ``timeout_setting`` names the :class:`~code_spider.config.Settings` field
    that governs this tool's wall-clock timeout. The generic default is
    ``tool_timeout_s`` (20 s); long-running tools such as ``index_repository``
    pass ``"index_timeout_s"`` so they are not killed by the generic cap.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            started = time.perf_counter()
            audit_kwargs = {k: v for k, v in kwargs.items() if k != "query_text"}
            timeout_s = _resolve_tool_timeout_s(timeout_setting)
            try:
                if timeout_s and timeout_s > 0:
                    future = _TOOL_EXECUTOR.submit(fn, *args, **kwargs)
                    try:
                        result = future.result(timeout=timeout_s)
                    except FuturesTimeoutError as exc:
                        # Don't block on the orphaned worker; it drains on its own.
                        future.cancel()
                        raise TimeoutError(
                            f"{tool_name} timed out after {timeout_s:g}s "
                            "(configure via CODE_SPIDER_TOOL_TIMEOUT_S)"
                        ) from exc
                else:
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
