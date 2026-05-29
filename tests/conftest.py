"""Shared fixtures for the Code Spider test suite.

Integration tests are marked with ``@pytest.mark.integration`` and are skipped
unless ``CODE_SPIDER_NEO4J_URI`` is reachable. Run them explicitly with::

    pytest -m integration
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from urllib.parse import urlparse

import pytest

from code_spider.config import Neo4jSettings


def _neo4j_settings_from_env() -> Neo4jSettings:
    return Neo4jSettings(
        uri=os.environ.get("CODE_SPIDER_NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("CODE_SPIDER_NEO4J_USER", "neo4j"),
        password=os.environ.get("CODE_SPIDER_NEO4J_PASSWORD", "codespider-dev-password"),
        database=os.environ.get("CODE_SPIDER_NEO4J_DATABASE", "neo4j"),
    )


def _neo4j_reachable(settings: Neo4jSettings) -> str | None:
    """Return None if Neo4j is up and credentials work, else a reason."""
    parsed = urlparse(settings.uri)
    host, port = parsed.hostname, parsed.port or 7687
    if not host:
        return "no hostname in uri"
    try:
        with socket.create_connection((host, port), timeout=1.5):
            pass
    except OSError as exc:
        return f"tcp connect failed: {exc}"
    # Validate auth via a trivial bolt session.
    try:
        from code_spider.graph import Neo4jClient

        with Neo4jClient(settings) as client, client.session() as session:
            session.run("RETURN 1").single()
    except Exception as exc:
        return f"bolt auth failed: {exc}"
    return None


@pytest.fixture(scope="session")
def neo4j_settings() -> Neo4jSettings:
    settings = _neo4j_settings_from_env()
    reason = _neo4j_reachable(settings)
    if reason:
        pytest.skip(f"Neo4j unavailable at {settings.uri}: {reason}")
    return settings


@pytest.fixture
def neo4j_client(neo4j_settings: Neo4jSettings) -> Iterator[object]:
    from code_spider.graph import Neo4jClient

    with Neo4jClient(neo4j_settings) as client:
        yield client
