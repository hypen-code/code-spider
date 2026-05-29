"""Thin wrapper around the neo4j-python driver.

Resilience strategy:

    - **Connection-level**: `:meth:`Neo4jClient.verify` retries on transient
      connection errors (``ServiceUnavailable``, ``ConnectionError``,
      timeouts) but never on auth/config errors — those must surface to the
      caller immediately so misconfiguration is loud.
    - **Transaction-level**: every batch goes through ``session.execute_write``
      / ``session.execute_read``, which the upstream driver already wraps in
      a retry loop on ``TransientError`` / ``SessionExpired``.
    - **High-level**: :func:`retry_on_transient` decorates top-level write
      operations (used by :class:`GraphWriter`) to recover from session pool
      hiccups during long multi-statement runs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from neo4j import Driver, GraphDatabase, ManagedTransaction, Session
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from code_spider.config import Neo4jSettings
from code_spider.logging_setup import get_logger

_log = get_logger(__name__)

# Errors we always retry. ``AuthError`` / ``ConfigurationError`` are explicitly
# *not* in this set so misconfiguration surfaces immediately to the operator.
_RETRIABLE: tuple[type[BaseException], ...] = (
    ServiceUnavailable,
    SessionExpired,
    TransientError,
    ConnectionError,
    TimeoutError,
)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    _log.warning(
        "neo4j retry",
        attempt=state.attempt_number,
        wait_s=round(getattr(state.next_action, "sleep", 0.0), 2)
        if state.next_action
        else 0.0,
        error=str(exc),
    )


def retry_on_transient[R](fn: Callable[..., R]) -> Callable[..., R]:
    """Decorator: retry ``fn`` on transient Neo4j errors with exponential backoff.

    Used by :class:`code_spider.graph.writer.GraphWriter` on top-level
    write methods to ride out session pool hiccups during long indexer runs.
    """
    decorated = retry(
        retry=retry_if_exception_type(_RETRIABLE),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10.0),
        before_sleep=_log_retry,
        reraise=True,
    )(fn)
    return decorated  # type: ignore[no-any-return]


class Neo4jClient:
    """Owns a single ``neo4j.Driver`` and exposes session/transaction helpers."""

    def __init__(self, settings: Neo4jSettings) -> None:
        self._settings = settings
        self._driver: Driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password),
            connection_timeout=10.0,
        )

    @retry(
        retry=retry_if_exception_type(_RETRIABLE),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        before_sleep=_log_retry,
        reraise=True,
    )
    def verify(self) -> None:
        """Block until Neo4j accepts a trivial query. Used by ``code-spider migrate``.

        Auth/configuration errors are never retried — they surface immediately.
        """
        try:
            with self.session() as session:
                session.run("RETURN 1 AS ok").single()
        except (AuthError, ConfigurationError) as exc:
            _log.error("neo4j auth/config failure", uri=self._settings.uri, error=str(exc))
            raise
        _log.info("neo4j connection verified", uri=self._settings.uri)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._driver.session(database=self._settings.database) as session:
            yield session

    def execute_write(self, work: Callable[..., Any], **kwargs: Any) -> Any:
        with self.session() as session:
            return session.execute_write(_wrap_work(work), **kwargs)

    def execute_read(self, work: Callable[..., Any], **kwargs: Any) -> Any:
        with self.session() as session:
            return session.execute_read(_wrap_work(work), **kwargs)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _wrap_work(work: Callable[..., Any]) -> Callable[[ManagedTransaction], Any]:
    """Adapter so callers can supply a function taking (tx, **kwargs)."""

    def runner(tx: ManagedTransaction, **kwargs: Any) -> Any:
        return work(tx, **kwargs)

    return runner
