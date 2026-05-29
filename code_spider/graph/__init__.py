"""Neo4j integration: client, schema migrations, idempotent writer, vector backends."""

from code_spider.graph.client import Neo4jClient
from code_spider.graph.schema import apply_schema
from code_spider.graph.writer import GraphWriter

__all__ = ["GraphWriter", "Neo4jClient", "apply_schema"]
