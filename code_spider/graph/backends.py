"""Vector backend abstraction.

Phase 0/1 ship the Neo4j HNSW implementation. Phase 2+ can drop in Qdrant
or pgvector by implementing the same protocol without touching agent code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    score: float


@runtime_checkable
class VectorBackend(Protocol):
    """Persistent vector index over code chunks."""

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
    ) -> None: ...

    def query(
        self,
        *,
        workspace_id: str,
        embedding: list[float],
        limit: int,
    ) -> list[VectorHit]: ...

    def delete_workspace(self, workspace_id: str) -> None: ...
