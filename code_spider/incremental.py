"""Incremental indexing — per-file BLAKE3 hash diff against Neo4j state.

Strategy:

    1. Walk the checkout and compute each candidate file's BLAKE3 hash.
    2. Query Neo4j for ``(path, hash_blake3)`` of every File node already
       indexed for ``(workspace_id, repo)``.
    3. Partition the working set into:
        - ``unchanged_paths``: hash matches → skip parse + skip write.
        - ``changed_paths``  : hash differs OR file is new.
        - ``deleted_paths``  : on graph, absent on disk → schedule deletion.

The indexer feeds the changed set through the normal parse/resolve/write
pipeline and asks the writer to perform a *surgical* update (no full repo
clear). The cross-service flow matchers always run workspace-wide so flow
edges remain consistent.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import blake3

from code_spider.graph.client import Neo4jClient
from code_spider.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class FileDiff:
    """Result of a per-repo file-hash diff."""

    workspace_id: str
    repo_name: str
    unchanged_paths: frozenset[str]
    changed_paths: frozenset[str]
    deleted_paths: frozenset[str]
    new_hashes: dict[str, str]  # changed_paths -> new BLAKE3 hex

    @property
    def total(self) -> int:
        return (
            len(self.unchanged_paths) + len(self.changed_paths) + len(self.deleted_paths)
        )

    @property
    def has_work(self) -> bool:
        return bool(self.changed_paths) or bool(self.deleted_paths)


_FILE_HASH_QUERY = """
MATCH (:Repository {workspace_id: $workspace_id, name: $repo_name})
      -[:HAS_COMMIT]->(:Commit)-[:CONTAINS]->(f:File)
RETURN f.path AS path, f.hash_blake3 AS hash
"""


def fetch_existing_hashes(
    *, client: Neo4jClient, workspace_id: str, repo_name: str
) -> dict[str, str]:
    """Return ``{path -> hash_blake3}`` for files currently in the graph."""
    with client.session() as session:
        result = session.run(
            _FILE_HASH_QUERY,
            workspace_id=workspace_id,
            repo_name=repo_name,
        )
        return {row["path"]: row["hash"] or "" for row in result if row["path"]}


def compute_diff(
    *,
    workspace_id: str,
    repo_name: str,
    existing_hashes: dict[str, str],
    on_disk_files: Iterable[tuple[str, bytes]],
) -> FileDiff:
    """Compute :class:`FileDiff` from on-disk ``(rel_path, source_bytes)`` pairs."""
    unchanged: set[str] = set()
    changed: set[str] = set()
    new_hashes: dict[str, str] = {}
    seen_paths: set[str] = set()

    for rel_path, source in on_disk_files:
        seen_paths.add(rel_path)
        new_hash = blake3.blake3(source).hexdigest()
        prior = existing_hashes.get(rel_path)
        if prior and prior == new_hash:
            unchanged.add(rel_path)
            continue
        changed.add(rel_path)
        new_hashes[rel_path] = new_hash

    deleted = set(existing_hashes.keys()) - seen_paths

    diff = FileDiff(
        workspace_id=workspace_id,
        repo_name=repo_name,
        unchanged_paths=frozenset(unchanged),
        changed_paths=frozenset(changed),
        deleted_paths=frozenset(deleted),
        new_hashes=new_hashes,
    )
    _log.info(
        "file diff computed",
        workspace=workspace_id,
        repo=repo_name,
        unchanged=len(unchanged),
        changed=len(changed),
        deleted=len(deleted),
    )
    return diff


def hash_file(path: Path) -> str:
    """Read ``path`` and return its hex BLAKE3 hash. Convenience helper."""
    return blake3.blake3(path.read_bytes()).hexdigest()
