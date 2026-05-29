"""Prometheus metrics for the indexer + MCP server.

Stays optional: Prometheus only emits when ``start_metrics_server`` is called
(usually from the CLI behind ``--metrics-port``). Otherwise the counters are
still updated in memory at zero cost — they're plain CollectorRegistry objects.

Metric naming follows the Prometheus best-practice ``<namespace>_<subsystem>_<name>_<unit>``
convention with stable labels (``stage``, ``workspace``, ``repo``, ``tool``).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    start_http_server,
)

from code_spider.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class _MetricsBundle:
    registry: CollectorRegistry
    indexer_stage_duration: Histogram
    indexer_files_parsed: Counter
    indexer_symbols_emitted: Counter
    indexer_resolved_calls: Counter
    indexer_http_flows: Counter
    indexer_kafka_flows: Counter
    indexer_chunks: Counter
    indexer_errors: Counter
    mcp_tool_duration: Histogram
    mcp_tool_errors: Counter


def _build() -> _MetricsBundle:
    reg = CollectorRegistry()
    return _MetricsBundle(
        registry=reg,
        indexer_stage_duration=Histogram(
            "code_spider_indexer_stage_duration_seconds",
            "Wall-clock duration of each indexer stage.",
            labelnames=("stage", "workspace"),
            registry=reg,
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300),
        ),
        indexer_files_parsed=Counter(
            "code_spider_indexer_files_parsed_total",
            "Files successfully parsed.",
            labelnames=("workspace", "repo", "lang"),
            registry=reg,
        ),
        indexer_symbols_emitted=Counter(
            "code_spider_indexer_symbols_emitted_total",
            "Symbols emitted to the graph.",
            labelnames=("workspace", "repo", "kind"),
            registry=reg,
        ),
        indexer_resolved_calls=Counter(
            "code_spider_indexer_resolved_calls_total",
            "Calls resolved by the cascade.",
            labelnames=("workspace", "strategy"),
            registry=reg,
        ),
        indexer_http_flows=Counter(
            "code_spider_indexer_http_flows_total",
            "HTTP_FLOW edges materialised.",
            labelnames=("workspace",),
            registry=reg,
        ),
        indexer_kafka_flows=Counter(
            "code_spider_indexer_kafka_flows_total",
            "KAFKA_FLOW edges materialised.",
            labelnames=("workspace",),
            registry=reg,
        ),
        indexer_chunks=Counter(
            "code_spider_indexer_chunks_total",
            "Chunks written to the graph.",
            labelnames=("workspace", "repo"),
            registry=reg,
        ),
        indexer_errors=Counter(
            "code_spider_indexer_errors_total",
            "Indexer errors by stage.",
            labelnames=("stage", "workspace"),
            registry=reg,
        ),
        mcp_tool_duration=Histogram(
            "code_spider_mcp_tool_duration_seconds",
            "Wall-clock duration of MCP tool invocations.",
            labelnames=("tool",),
            registry=reg,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        ),
        mcp_tool_errors=Counter(
            "code_spider_mcp_tool_errors_total",
            "MCP tool invocation errors.",
            labelnames=("tool",),
            registry=reg,
        ),
    )


METRICS: _MetricsBundle = _build()


@contextmanager
def stage_timer(stage: str, workspace: str) -> Iterator[None]:
    """Context manager that records the wall-clock duration of an indexer stage."""
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        METRICS.indexer_errors.labels(stage=stage, workspace=workspace).inc()
        raise
    finally:
        elapsed = time.perf_counter() - started
        METRICS.indexer_stage_duration.labels(stage=stage, workspace=workspace).observe(elapsed)


def start_metrics_server(port: int = 9464) -> None:
    """Start an HTTP server exposing the Prometheus metrics endpoint."""
    start_http_server(port, registry=METRICS.registry)
    _log.info("prometheus metrics server started", port=port, path="/metrics")
