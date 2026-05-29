"""Lexical search over the ``symbol_text`` fulltext index."""

from __future__ import annotations

from dataclasses import dataclass

from code_spider.graph.client import Neo4jClient


@dataclass(frozen=True, slots=True)
class LexicalHit:
    workspace_id: str
    repo: str
    fqn: str
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    score: float


def lexical_search(
    *,
    client: Neo4jClient,
    workspace_id: str,
    query: str,
    limit: int = 20,
) -> list[LexicalHit]:
    """Query the ``symbol_text`` fulltext index and filter by workspace.

    Returns Symbol hits ordered by Lucene relevance score (descending).
    """
    if not query.strip():
        return []
    with client.session() as session:
        result = session.run(
            """
            CALL db.index.fulltext.queryNodes('symbol_text', $query)
            YIELD node, score
            WITH node, score
            WHERE node.workspace_id = $workspace_id
            RETURN node.workspace_id AS workspace_id,
                   node.repo         AS repo,
                   node.fqn          AS fqn,
                   node.name         AS name,
                   node.kind         AS kind,
                   node.file_path    AS file_path,
                   node.start_line   AS start_line,
                   node.end_line     AS end_line,
                   score             AS score
            ORDER BY score DESC
            LIMIT $limit
            """,
            workspace_id=workspace_id,
            query=query,
            limit=limit,
        )
        return [
            LexicalHit(
                workspace_id=r["workspace_id"],
                repo=r["repo"],
                fqn=r["fqn"],
                name=r["name"],
                kind=r["kind"],
                file_path=r["file_path"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                score=float(r["score"]),
            )
            for r in result
        ]
