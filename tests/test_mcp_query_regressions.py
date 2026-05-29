"""Regression tests for two MCP-tool Neo4j query bugs:

1. ``semantic_code_search`` (→ ``lexical_search``) used to pass the Cypher
   parameter ``query=`` as a ``**kwparameter`` to ``Session.run``, colliding
   with the driver's own positional ``query`` argument (raises in
   ``neo4j-driver`` >=6.0).
2. ``get_call_graph`` used ``length(c1)`` / ``length(c2)`` on a relationship
   list bound by a variable-length pattern. Cypher's ``length()`` only
   accepts a Path; for lists you must use ``size()``.

These tests exercise the *exact* offending code paths with a fake Neo4j
session so they protect against silent regressions even when no live
database is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

# NOTE: ``code_spider.mcp.tools.get_call_graph`` resolves to the function (the
# parent package's ``__init__.py`` re-exports it), so we import the Cypher
# template directly from the submodule.
from code_spider.mcp.tools.get_call_graph import _QUERY as GET_CALL_GRAPH_QUERY
from code_spider.search.lexical import lexical_search

# --------------------------------------------------------------------------- #
# Fakes that mimic the parts of the neo4j driver we use.                      #
# --------------------------------------------------------------------------- #


@dataclass
class _StrictRun:
    """Records and validates how ``Session.run`` was invoked.

    Mimics neo4j-driver >=6.0 behaviour: any ``**kwparameter`` with the
    reserved name ``query`` raises a TypeError, exactly like the real driver.
    """

    last_query: str | None = None
    last_parameters: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwparameters: Any,
    ) -> _StrictRun:
        # Real driver: kwparameter names cannot shadow the positional ``query``.
        if "query" in kwparameters:
            raise TypeError(
                "Session.run() got multiple values for argument 'query'"
            )
        self.last_query = query
        merged: dict[str, Any] = {}
        if parameters:
            merged.update(parameters)
        merged.update(kwparameters)
        self.last_parameters = merged
        return self

    # Iteration over a result yields dict-like records.
    def __iter__(self):
        return iter(self.rows)


class _FakeSession:
    def __init__(self, runner: _StrictRun) -> None:
        self._runner = runner

    def __enter__(self) -> _StrictRun:
        return self._runner

    def __exit__(self, *exc: Any) -> None:
        return None


class _FakeClient:
    def __init__(self, runner: _StrictRun) -> None:
        self._runner = runner

    def session(self) -> _FakeSession:
        return _FakeSession(self._runner)


# --------------------------------------------------------------------------- #
# Bug 1 — lexical_search must NOT pass ``query`` as a kwparameter.            #
# --------------------------------------------------------------------------- #


def test_lexical_search_does_not_collide_with_session_run_query_kw() -> None:
    runner = _StrictRun(rows=[])
    client = _FakeClient(runner)

    # Must not raise. Pre-fix, this raised TypeError because lexical_search
    # called ``session.run(cypher, query=query, ...)``.
    hits = lexical_search(
        client=client,  # type: ignore[arg-type]
        workspace_id="demo",
        query="add_user",
        limit=5,
    )

    assert hits == []
    # The user-supplied search term must still arrive as Cypher parameter ``$query``.
    assert runner.last_parameters is not None
    assert runner.last_parameters["query"] == "add_user"
    assert runner.last_parameters["workspace_id"] == "demo"
    assert runner.last_parameters["limit"] == 5


def test_strict_run_fake_raises_on_query_kwparameter() -> None:
    """Sanity-check that our fake actually catches the bug we are guarding
    against — otherwise the regression test above would be a no-op."""
    runner = _StrictRun()
    with pytest.raises(TypeError, match="multiple values"):
        runner.run("MATCH (n) RETURN n", query="oops")


# --------------------------------------------------------------------------- #
# Bug 2 — get_call_graph Cypher must use size() not length() on rel-lists.    #
# --------------------------------------------------------------------------- #


def test_get_call_graph_query_uses_size_not_length_on_rel_lists() -> None:
    cypher = GET_CALL_GRAPH_QUERY
    # The relationship lists from variable-length CALLS patterns are named
    # ``c1`` and ``c2``. ``length(<list>)`` is illegal in Neo4j 5+; ``size``
    # is the correct collection cardinality function.
    assert "size(c1)" in cypher
    assert "size(c2)" in cypher
    assert "length(c1)" not in cypher
    assert "length(c2)" not in cypher


def test_get_call_graph_query_template_renders_with_depth() -> None:
    rendered = GET_CALL_GRAPH_QUERY % {"depth": 3}
    # Depth interpolation still works after the size() fix.
    assert "CALLS*1..3" in rendered
    # No stray ``%(`` placeholders left behind.
    assert "%(depth)" not in rendered
