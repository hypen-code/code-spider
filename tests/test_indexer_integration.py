"""End-to-end Phase 0 smoke test: index a fixture repo into a live Neo4j.

Requires a reachable Neo4j (Community 5.13+) — skipped automatically otherwise.
Run with::

    pytest -m integration
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from code_spider.config import Settings
from code_spider.graph import Neo4jClient, apply_schema
from code_spider.indexer import index_workspace
from code_spider.workspace.manifest import Manifest, RepoConfig, WorkspaceConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fixture-repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "calc.py").write_text(
        dedent(
            '''
            """Tiny calc module."""

            def add(a: int, b: int) -> int:
                """Add two ints."""
                return a + b


            class Multiplier:
                def __init__(self, factor: int) -> None:
                    self.factor = factor

                def apply(self, value: int) -> int:
                    return value * self.factor
            '''
        ),
        encoding="utf-8",
    )
    return repo


def test_phase0_index_writes_expected_graph(
    fixture_repo: Path, neo4j_settings, tmp_path: Path
) -> None:
    manifest = Manifest(
        version=1,
        workspaces=[
            WorkspaceConfig(
                id="pytest_demo",
                name="Pytest Demo",
                repos=[
                    RepoConfig(
                        name="fixture",
                        path=str(fixture_repo),
                        branch="main",
                        languages=["python"],
                    )
                ],
            )
        ],
    )

    settings = Settings(
        neo4j=neo4j_settings,
        manifest_path=tmp_path / "workspaces.yaml",
        checkout_root=tmp_path / "checkouts",
        log_level="WARNING",
        log_json=False,
    )

    with Neo4jClient(neo4j_settings) as client:
        apply_schema(client)
        # Clean up any prior pytest run.
        with client.session() as session:
            session.run(
                "MATCH (n {workspace_id: 'pytest_demo'}) DETACH DELETE n"
            ).consume()

    results = index_workspace(
        manifest=manifest,
        workspace_id="pytest_demo",
        settings=settings,
    )
    assert results, "indexer returned no results"

    with Neo4jClient(neo4j_settings) as client, client.session() as session:
        counts = session.run(
            """
            MATCH (w:Workspace {id: 'pytest_demo'})
            OPTIONAL MATCH (w)-[:CONTAINS]->(r:Repository)
            OPTIONAL MATCH (r)-[:HAS_COMMIT]->(c:Commit)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(s:Symbol)
            RETURN
              count(DISTINCT r) AS repos,
              count(DISTINCT f) AS files,
              count(DISTINCT s) AS symbols
            """
        ).single()
        assert counts is not None
        assert counts["repos"] == 1
        assert counts["files"] >= 2  # __init__.py + calc.py
        assert counts["symbols"] >= 4  # add, Multiplier, __init__, apply

        kinds = session.run(
            """
            MATCH (s:Symbol {workspace_id: 'pytest_demo'})
            RETURN s.kind AS kind, count(*) AS n
            """
        ).data()
        kind_map = {row["kind"]: row["n"] for row in kinds}
        assert kind_map.get("class", 0) >= 1
        assert kind_map.get("method", 0) >= 2
        assert kind_map.get("function", 0) >= 1
