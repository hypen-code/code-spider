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


# Default embedding dimension matches sentence-transformers/all-MiniLM-L6-v2.
# The effective dimension is read from ``CODE_SPIDER_EMBED_DIM`` (via
# :class:`code_spider.config.EmbeddingSettings`) and may be overridden when
# switching to an external model (e.g. ``voyage-code-3`` is 1024 dim,
# ``text-embedding-3-small`` is 1536 dim). Changing this value requires
# re-running ``code-spider migrate`` AND re-indexing all chunks.
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


def _vector_index(dim: int) -> str:
    return f"""
    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
    FOR (c:Chunk) ON c.embedding
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {dim},
      `vector.similarity_function`: 'cosine'
    }}}}
    """


def _existing_vector_dim(client: Neo4jClient) -> int | None:
    """Return the configured ``vector.dimensions`` of an existing index, or ``None``.

    We pre-check so the operator gets a clear error instead of a confusing
    "dimension mismatch" failure deep in the indexer when they switch models
    without re-creating the index.
    """
    cypher = (
        "SHOW VECTOR INDEXES YIELD name, options "
        "WHERE name = 'chunk_embedding' "
        "RETURN options.indexConfig.`vector.dimensions` AS dim"
    )
    try:
        with client.session() as session:
            row = session.run(cypher).single()
    except Exception:
        # Older Neo4j versions lack SHOW VECTOR INDEXES; fall back silently
        # and let ``CREATE ... IF NOT EXISTS`` handle the apply step.
        return None
    if row is None:
        return None
    value = row["dim"]
    return int(value) if value is not None else None


def apply_schema(
    client: Neo4jClient,
    *,
    embedding_dim: int = EMBEDDING_DIM,
) -> None:
    """Apply all constraints + indexes. Idempotent.

    When the existing ``chunk_embedding`` index has a different vector
    dimension than ``embedding_dim``, the index is automatically dropped and
    recreated at the new dimension. **This deletes every existing chunk
    embedding** — the caller must reindex affected workspaces afterwards.
    Neo4j's ``CREATE ... IF NOT EXISTS`` would silently keep the old
    definition, which would later cause confusing dim-mismatch errors at
    write time; auto-adjusting here keeps ``CODE_SPIDER_EMBED_DIM`` as the
    single source of truth.

    Args:
        client: Open :class:`Neo4jClient`.
        embedding_dim: Vector dimension to use for the ``chunk_embedding``
            index. Pass ``settings.embedding.dim`` from a real ``Settings``
            instance to honour the operator's ``CODE_SPIDER_EMBED_DIM`` env.
    """
    existing = _existing_vector_dim(client)
    if existing is not None and existing != embedding_dim:
        _log.warning(
            "chunk_embedding dimension changed; dropping and recreating index "
            "(all existing chunk embeddings will be removed — reindex affected "
            "workspaces afterwards)",
            old_dim=existing,
            new_dim=embedding_dim,
        )
        with client.session() as session:
            session.run("DROP INDEX chunk_embedding IF EXISTS").consume()

    statements: tuple[str, ...] = (
        *_CONSTRAINTS,
        *_BTREE_INDEXES,
        *_FULLTEXT_INDEXES,
        _vector_index(embedding_dim),
    )
    with client.session() as session:
        for stmt in statements:
            session.run(stmt).consume()
            _log.debug("applied schema statement", stmt=_oneline(stmt))
    _log.info("schema applied", statements=len(statements), embedding_dim=embedding_dim)


def _oneline(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())
