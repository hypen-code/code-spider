"""In-process TTL cache for ``get_graph_schema`` per-label / per-rel counts.

The schema tool runs one Cypher round-trip per label plus one per relationship
type when ``include_counts=True``. On a real workspace that is ~25 queries
(~250-400 ms total). The vast majority of ``get_graph_schema`` calls come from
LLM agents about to issue ``execute_cypher`` — and they typically issue several
queries in a row. Caching the count rows for a short TTL collapses the second
and subsequent schema fetches from ~300 ms to <10 ms without changing semantics.

Design choices
--------------
* **In-process only.** Multi-replica MCP deployments will see a per-replica
  view of staleness up to ``ttl_s`` seconds. We document this and keep the
  default TTL short (60 s) so it is bounded.
* **Active invalidation** on every write through :mod:`code_spider.graph.writer`
  so a fresh index/incremental delta is reflected immediately within the
  same process.
* **Thread-safe** via a single :class:`threading.RLock` — the schema tool can
  be called concurrently by FastMCP and we never want two threads racing on
  the same key.
* **Bypass-able** via ``force_refresh=True`` on the public tool so an agent
  can always demand a fresh snapshot.
* **Metrics-instrumented** so operators can spot abnormal hit/miss ratios.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from code_spider.logging_setup import get_logger
from code_spider.observability import METRICS

_log = get_logger(__name__)


# ----------------------------- Configuration ------------------------------ #


_DEFAULT_TTL_S: float = 60.0
_ENV_TTL = "CODE_SPIDER_SCHEMA_COUNT_TTL_S"


def _default_ttl() -> float:
    """Resolve the default TTL from the environment, with safe fallback."""
    raw = os.getenv(_ENV_TTL)
    if not raw:
        return _DEFAULT_TTL_S
    try:
        value = float(raw)
    except ValueError:
        _log.warning(
            "invalid count-cache TTL env value; falling back to default",
            env=_ENV_TTL,
            value=raw,
            default=_DEFAULT_TTL_S,
        )
        return _DEFAULT_TTL_S
    if value < 0:
        return 0.0
    return value


# ------------------------------- Cache types ------------------------------ #


Kind = Literal["node", "rel"]
Scope = Literal["workspace", "global"]


@dataclass(frozen=True, slots=True)
class CountEntry:
    """A single cached count row."""

    value: int
    scope: Scope


@dataclass(frozen=True, slots=True)
class _CacheRecord:
    entry: CountEntry
    expires_at: float


def _make_key(
    *, workspace_id: str | None, kind: Kind, name: str, scoped: bool
) -> tuple[str | None, Kind, str, bool]:
    """Key shape: ``(workspace_id, kind, label_or_reltype, scoped)``.

    ``scoped`` is part of the key because the same label can be cached
    under both global and workspace scopes (different Cypher, different
    answers).
    """
    return (workspace_id, kind, name, scoped)


# --------------------------------- Cache ---------------------------------- #


class CountCache:
    """Thread-safe TTL cache for per-label / per-rel-type counts."""

    def __init__(self, ttl_s: float | None = None) -> None:
        self._ttl = _default_ttl() if ttl_s is None else max(0.0, float(ttl_s))
        self._store: dict[tuple[str | None, Kind, str, bool], _CacheRecord] = {}
        self._lock = threading.RLock()

    # --- Introspection -------------------------------------------------- #

    @property
    def ttl_s(self) -> float:
        return self._ttl

    def set_ttl(self, ttl_s: float) -> None:
        """Override the TTL at runtime (mainly for tests)."""
        with self._lock:
            self._ttl = max(0.0, float(ttl_s))

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Drop every entry. Used by tests and by an admin endpoint later."""
        with self._lock:
            self._store.clear()

    # --- Core read/write ------------------------------------------------ #

    def get_or_compute(
        self,
        *,
        workspace_id: str | None,
        kind: Kind,
        name: str,
        scoped: bool,
        fetch: Callable[[], CountEntry],
        force_refresh: bool = False,
    ) -> CountEntry:
        """Return a cached count if fresh, otherwise call ``fetch`` and cache.

        ``fetch`` must return a :class:`CountEntry`. Exceptions raised by
        ``fetch`` propagate; nothing is cached on failure.

        TTL of 0 disables caching entirely (every call is a miss).
        """
        key = _make_key(workspace_id=workspace_id, kind=kind, name=name, scoped=scoped)
        now = time.monotonic()

        if not force_refresh and self._ttl > 0:
            with self._lock:
                rec = self._store.get(key)
                if rec is not None and rec.expires_at > now:
                    METRICS.schema_count_cache_hits.labels(kind=kind).inc()
                    return rec.entry

        METRICS.schema_count_cache_misses.labels(kind=kind).inc()
        entry = fetch()

        if self._ttl > 0:
            with self._lock:
                self._store[key] = _CacheRecord(
                    entry=entry, expires_at=time.monotonic() + self._ttl
                )
        return entry

    # --- Invalidation --------------------------------------------------- #

    def invalidate(
        self, workspace_id: str | None, *, trigger: str = "manual"
    ) -> int:
        """Drop every entry tied to ``workspace_id``.

        Also drops entries whose scope is global (``workspace_id is None``)
        because a write may have added or removed nodes that affect the
        global counts.

        Returns the number of entries dropped (for tests + observability).
        """
        dropped = 0
        with self._lock:
            for key in list(self._store.keys()):
                cached_ws = key[0]
                if cached_ws == workspace_id or cached_ws is None:
                    self._store.pop(key, None)
                    dropped += 1
        if dropped:
            METRICS.schema_count_cache_invalidations.labels(trigger=trigger).inc()
            _log.info(
                "schema count cache invalidated",
                workspace=workspace_id,
                trigger=trigger,
                dropped=dropped,
            )
        return dropped


# Process-wide singleton. Tests can construct their own ``CountCache`` and
# swap it in via :func:`set_cache` if they need a clean room.
_CACHE: CountCache = CountCache()


def get_cache() -> CountCache:
    return _CACHE


def set_cache(cache: CountCache) -> None:
    """Replace the module-level singleton (test hook)."""
    global _CACHE
    _CACHE = cache


def invalidate_workspace(workspace_id: str | None, *, trigger: str = "manual") -> int:
    """Convenience wrapper used by the writer to invalidate on writes."""
    return _CACHE.invalidate(workspace_id, trigger=trigger)
