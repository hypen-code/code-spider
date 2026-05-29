"""Plain dataclasses describing the in-memory graph payload.

These are the *internal* representation produced by language adapters and
consumed by :mod:`code_spider.graph.writer`. They mirror but do not duplicate
the Neo4j schema; the writer maps these into Cypher MERGEs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SymbolKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    VARIABLE = "variable"


@dataclass(frozen=True, slots=True)
class Span:
    """Source range — 1-indexed lines, 0-indexed columns (Tree-sitter convention)."""

    start_line: int
    start_col: int
    end_line: int
    end_col: int


@dataclass(frozen=True, slots=True)
class Symbol:
    """A named definition extracted from source."""

    fqn: str
    name: str
    kind: SymbolKind
    lang: str
    file_path: str
    span: Span
    signature: str = ""
    docstring: str = ""
    visibility: str = "public"  # public | private | protected (heuristic by leading "_")
    parent_fqn: str | None = None  # enclosing class/function FQN, if any


@dataclass(frozen=True, slots=True)
class Module:
    """A logical Python/JS module or TS namespace."""

    fqn: str
    kind: str  # "package" | "module"
    file_path: str | None  # None for namespace packages


@dataclass(frozen=True, slots=True)
class Import:
    """An import statement. ``raw`` is the literal source text; ``resolved_fqn``
    is filled by the resolver in Phase 1 — empty in Phase 0."""

    file_path: str
    raw: str
    local_name: str
    target_fqn: str
    span: Span
    resolved_fqn: str | None = None


@dataclass(frozen=True, slots=True)
class CallSite:
    """An unresolved call-expression reference. Resolved into a `:CALLS` edge later."""

    caller_fqn: str
    file_path: str
    call_text: str  # raw text of the callee expression
    span: Span


@dataclass(frozen=True, slots=True)
class Route:
    """A REST/HTTP route handler discovered on the provider side (Phase 1)."""

    method: str
    path: str
    framework: str
    handler_fqn: str
    file_path: str
    span: Span


@dataclass(frozen=True, slots=True)
class HttpClientCall:
    """An HTTP client call on the consumer side (Phase 1)."""

    caller_fqn: str
    method: str
    path_template: str
    base_url_hint: str | None
    file_path: str
    span: Span


@dataclass(frozen=True, slots=True)
class KafkaTopic:
    """Logical Kafka topic shared across the workspace (Phase 1)."""

    name: str
    cluster_hint: str | None = None


@dataclass(frozen=True, slots=True)
class KafkaProducer:
    """A code site that publishes to a Kafka topic (Phase 1)."""

    caller_fqn: str
    topic_name: str
    client_lib: str
    file_path: str
    span: Span


@dataclass(frozen=True, slots=True)
class KafkaConsumer:
    """A code site that consumes from a Kafka topic (Phase 1)."""

    caller_fqn: str
    topic_name: str
    client_lib: str
    group_id: str | None
    file_path: str
    span: Span


@dataclass(frozen=True, slots=True)
class Chunk:
    """An AST-aware code chunk for hybrid search (Phase 1)."""

    chunk_id: str
    file_path: str
    span: Span
    text: str
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedCall:
    """A CallSite mapped to a concrete callee Symbol by the resolver cascade."""

    caller_fqn: str
    callee_fqn: str
    callee_repo: str  # repo where the callee lives (may differ from caller)
    confidence: float
    strategy: str
    file_path: str
    span: Span


@dataclass(frozen=True, slots=True)
class HttpFlowEdge:
    """Cross-service producer→provider edge linking a client call to a route."""

    client_caller_fqn: str
    client_repo: str
    client_file_path: str
    client_span: Span
    route_handler_fqn: str
    route_repo: str
    method: str
    path_template: str
    match_score: float


@dataclass(frozen=True, slots=True)
class KafkaFlowEdge:
    """Producer→consumer edge materialised via a shared KafkaTopic."""

    producer_caller_fqn: str
    producer_repo: str
    consumer_caller_fqn: str
    consumer_repo: str
    topic_name: str


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One file's full parse result, ready for upsert."""

    repo_relative_path: str
    lang: str
    hash_blake3: str
    size_bytes: int
    line_count: int
    module: Module | None
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    call_sites: list[CallSite] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    http_clients: list[HttpClientCall] = field(default_factory=list)
    kafka_producers: list[KafkaProducer] = field(default_factory=list)
    kafka_consumers: list[KafkaConsumer] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Aggregate parse output for an entire repo at a given commit."""

    workspace_id: str
    repo_name: str
    commit_sha: str
    files: list[FileRecord]

    # Populated after the cross-repo passes (resolver, HTTP_FLOW matcher,
    # KAFKA_FLOW materialiser). Empty until those stages run.
    resolved_calls: list[ResolvedCall] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WorkspaceParseBundle:
    """Aggregated parse output for an entire workspace at one indexer run.

    Cross-repo edges (HTTP_FLOW, KAFKA_FLOW, cross-repo CALLS) live here
    because they cannot be attributed to a single repo.
    """

    workspace_id: str
    workspace_name: str
    manifest_sha: str
    repos: list[ParseResult] = field(default_factory=list)
    http_flows: list[HttpFlowEdge] = field(default_factory=list)
    kafka_flows: list[KafkaFlowEdge] = field(default_factory=list)
