"""Neo4j-backed implementation of :class:`code_spider.graph.backends.VectorBackend`.

Uses the native HNSW vector index created by
:func:`code_spider.graph.schema.apply_schema`. All operations are scoped by
``workspace_id`` so multiple tenants can share one Neo4j instance.
"""

from __future__ import annotations

from code_spider.graph.backends import VectorHit
from code_spider.graph.client import Neo4jClient


class Neo4jVectorBackend:
    """``VectorBackend`` implementation that stores vectors on ``:Chunk`` nodes."""

    name: str = "neo4j"

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    def upsert(
        self,
        *,
        workspace_id: str,
        chunk_id: str,
        file_path: str,
        start_line: int,
        end_line: int,
        text: str,
        embedding: list[float],
    ) -> None:
        with self._client.session() as session:
            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.workspace_id = $workspace_id,
                    c.file_path = $file_path,
                    c.start_line = $start_line,
                    c.end_line = $end_line,
                    c.text = $text,
                    c.embedding = $embedding
                """,
                chunk_id=chunk_id,
                workspace_id=workspace_id,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                text=text,
                embedding=embedding,
            ).consume()

    def query(
        self,
        *,
        workspace_id: str,
        embedding: list[float],
        limit: int,
    ) -> list[VectorHit]:
        with self._client.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('chunk_embedding', $limit, $vec)
                YIELD node, score
                WITH node, score
                WHERE node.workspace_id = $workspace_id
                RETURN node.chunk_id     AS chunk_id,
                       node.file_path    AS file_path,
                       node.start_line   AS start_line,
                       node.end_line     AS end_line,
                       score             AS score
                ORDER BY score DESC
                """,
                workspace_id=workspace_id,
                vec=embedding,
                limit=limit * 4,  # over-fetch then filter by workspace
            )
            hits = [
                VectorHit(
                    chunk_id=record["chunk_id"],
                    file_path=record["file_path"],
                    start_line=record["start_line"],
                    end_line=record["end_line"],
                    score=float(record["score"]),
                )
                for record in result
            ]
            return hits[:limit]

    def delete_workspace(self, workspace_id: str) -> None:
        with self._client.session() as session:
            session.run(
                "MATCH (c:Chunk {workspace_id: $workspace_id}) DETACH DELETE c",
                workspace_id=workspace_id,
            ).consume()
