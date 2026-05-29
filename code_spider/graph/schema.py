"""Graph schema migrations: constraints + fulltext + vector indexes.

Idempotent — every statement is ``CREATE ... IF NOT EXISTS``. Safe to run on
every indexer cold start.

Compatible with **Neo4j 5.13+ Community Edition** (required for native vector
indexes). RBAC / fine-grained access control is not used; auth is basic and
relies on separate Neo4j users (indexer rw, MCP ro) — see the design plan.
"""

from __future__ import annotations

from code_spider.graph.client import Neo4jClient
from code_spider.logging_setup import get_logger

_log = get_logger(__name__)


# Embedding dimension matches sentence-transformers/all-MiniLM-L6-v2 (Phase 1
# default). Changing this requires re-indexing all chunks.
EMBEDDING_DIM = 384


_CONSTRAINTS: tuple[str, ...] = (
    """CREATE CONSTRAINT workspace_id IF NOT EXISTS
       FOR (w:Workspace) REQUIRE w.id IS UNIQUE""",
    """CREATE CONSTRAINT repository_key IF NOT EXISTS
       FOR (r:Repository) REQUIRE (r.workspace_id, r.name) IS UNIQUE""",
    """CREATE CONSTRAINT commit_key IF NOT EXISTS
       FOR (c:Commit) REQUIRE (c.workspace_id, c.repo, c.sha) IS UNIQUE""",
    """CREATE CONSTRAINT file_key IF NOT EXISTS
       FOR (f:File) REQUIRE (f.workspace_id, f.repo, f.path) IS UNIQUE""",
    """CREATE CONSTRAINT module_key IF NOT EXISTS
       FOR (m:Module) REQUIRE (m.workspace_id, m.repo, m.fqn) IS UNIQUE""",
    """CREATE CONSTRAINT symbol_key IF NOT EXISTS
       FOR (s:Symbol) REQUIRE (s.workspace_id, s.repo, s.fqn) IS UNIQUE""",
    """CREATE CONSTRAINT chunk_key IF NOT EXISTS
       FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE""",
    """CREATE CONSTRAINT kafka_topic_key IF NOT EXISTS
       FOR (t:KafkaTopic) REQUIRE (t.workspace_id, t.name) IS UNIQUE""",
    """CREATE CONSTRAINT route_key IF NOT EXISTS
       FOR (r:Route) REQUIRE r.route_id IS UNIQUE""",
    """CREATE CONSTRAINT http_client_key IF NOT EXISTS
       FOR (h:HttpClientCall) REQUIRE h.call_id IS UNIQUE""",
    """CREATE CONSTRAINT kafka_producer_key IF NOT EXISTS
       FOR (p:KafkaProducer) REQUIRE p.producer_id IS UNIQUE""",
    """CREATE CONSTRAINT kafka_consumer_key IF NOT EXISTS
       FOR (c:KafkaConsumer) REQUIRE c.consumer_id IS UNIQUE""",
)


_BTREE_INDEXES: tuple[str, ...] = (
    """CREATE INDEX symbol_name_idx IF NOT EXISTS
       FOR (s:Symbol) ON (s.name)""",
    """CREATE INDEX file_lang_idx IF NOT EXISTS
       FOR (f:File) ON (f.lang)""",
    """CREATE INDEX route_method_path_idx IF NOT EXISTS
       FOR (r:Route) ON (r.method, r.path)""",
    """CREATE INDEX chunk_workspace_idx IF NOT EXISTS
       FOR (c:Chunk) ON (c.workspace_id)""",
    """CREATE INDEX kafka_producer_workspace_idx IF NOT EXISTS
       FOR (p:KafkaProducer) ON (p.workspace_id, p.repo)""",
    """CREATE INDEX kafka_consumer_workspace_idx IF NOT EXISTS
       FOR (c:KafkaConsumer) ON (c.workspace_id, c.repo)""",
)


_FULLTEXT_INDEXES: tuple[str, ...] = (
    """CREATE FULLTEXT INDEX symbol_text IF NOT EXISTS
       FOR (s:Symbol) ON EACH [s.name, s.signature, s.docstring]""",
)


def _vector_index() -> str:
    return f"""
    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
    FOR (c:Chunk) ON c.embedding
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {EMBEDDING_DIM},
      `vector.similarity_function`: 'cosine'
    }}}}
    """


def apply_schema(client: Neo4jClient) -> None:
    """Apply all constraints + indexes. Idempotent."""
    statements: tuple[str, ...] = (
        *_CONSTRAINTS,
        *_BTREE_INDEXES,
        *_FULLTEXT_INDEXES,
        _vector_index(),
    )
    with client.session() as session:
        for stmt in statements:
            session.run(stmt).consume()
            _log.debug("applied schema statement", stmt=_oneline(stmt))
    _log.info("schema applied", statements=len(statements))


def _oneline(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())
