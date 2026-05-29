"""Unit tests for ``execute_cypher`` — the ad-hoc read-only Cypher tool.

These tests exercise the three layers of read-only enforcement plus the
input validators, all without touching a live Neo4j. A fake session
captures every ``Session.run`` invocation and mimics the driver enough
to drive both the EXPLAIN pre-flight and the actual transactional run.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any
from unittest.mock import patch

import pytest

# NOTE: ``code_spider.mcp.tools.execute_cypher`` is re-bound to the *function*
# by the parent package's ``__init__``. Pull the actual *module* via
# importlib so ``patch.object(mod, ...)`` targets the right namespace.
ec_mod = importlib.import_module("code_spider.mcp.tools.execute_cypher")
DEFAULT_LIMIT = ec_mod.DEFAULT_LIMIT
MAX_LIMIT = ec_mod.MAX_LIMIT
MAX_QUERY_CHARS = ec_mod.MAX_QUERY_CHARS
_check_no_writes = ec_mod._check_no_writes
_check_single_statement = ec_mod._check_single_statement
_scrub = ec_mod._scrub
_serialize_value = ec_mod._serialize_value
_validate_parameters = ec_mod._validate_parameters
execute_cypher = ec_mod.execute_cypher


# --------------------------------------------------------------------------- #
# 1. Pure-function safety checks                                              #
# --------------------------------------------------------------------------- #


class TestScrubAndKeywordScan:
    def test_scrub_strips_single_and_double_quoted_strings(self) -> None:
        cypher = (
            "MATCH (n) WHERE n.name = 'CREATE foo' OR n.note = \"DELETE bar\" "
            "RETURN n"
        )
        scrubbed = _scrub(cypher)
        assert "CREATE foo" not in scrubbed
        assert "DELETE bar" not in scrubbed

    def test_scrub_strips_line_and_block_comments(self) -> None:
        cypher = """
        // CREATE (n)
        /* MERGE (m)
           DELETE x */
        MATCH (n) RETURN n
        """
        scrubbed = _scrub(cypher)
        assert "CREATE" not in scrubbed
        assert "MERGE" not in scrubbed
        assert "DELETE" not in scrubbed
        assert "MATCH" in scrubbed

    def test_scrub_strips_backticked_identifiers(self) -> None:
        # A pathological label/property name that *spells* a write keyword.
        cypher = "MATCH (n:`CREATE`) RETURN n.`DELETE`"
        scrubbed = _scrub(cypher)
        assert "CREATE" not in scrubbed
        assert "DELETE" not in scrubbed

    @pytest.mark.parametrize(
        "bad_query",
        [
            "CREATE (n:Foo) RETURN n",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "MERGE (n:Foo {id: 1}) RETURN n",
            "MATCH (n) SET n.x = 1",
            "MATCH (n) REMOVE n.x",
            "DROP CONSTRAINT foo",
            "ALTER USER neo4j SET PASSWORD 'x'",
            "LOAD CSV FROM 'file:///x' AS row RETURN row",
            "USING PERIODIC COMMIT MATCH (n) RETURN n",
        ],
    )
    def test_static_scan_flags_obvious_writes(self, bad_query: str) -> None:
        with pytest.raises(PermissionError, match="not allowed"):
            _check_no_writes(bad_query)

    @pytest.mark.parametrize(
        "good_query",
        [
            "MATCH (n) RETURN n",
            "MATCH (n) WHERE n.name = 'CREATE foo' RETURN n",
            "// CREATE (x)\nMATCH (n) RETURN n",
            "/* MERGE x */ MATCH (n) RETURN n",
            "MATCH (n:`CREATE`) RETURN n",
            "CALL db.labels() YIELD label RETURN label",
            "MATCH p=(a)-[*1..3]->(b) RETURN p LIMIT 10",
        ],
    )
    def test_static_scan_passes_reads_and_false_positive_traps(
        self, good_query: str
    ) -> None:
        _check_no_writes(good_query)  # must not raise

    def test_keyword_is_reported_uppercased(self) -> None:
        with pytest.raises(PermissionError, match="'CREATE'"):
            _check_no_writes("create (n) return n")


class TestSingleStatementCheck:
    def test_trailing_semicolon_is_fine(self) -> None:
        _check_single_statement("MATCH (n) RETURN n;")
        _check_single_statement("MATCH (n) RETURN n;  ")

    def test_internal_semicolon_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="single Cypher statement"):
            _check_single_statement("MATCH (n) RETURN n; MATCH (m) RETURN m")

    def test_semicolon_inside_string_is_ignored(self) -> None:
        _check_single_statement(
            "MATCH (n) WHERE n.note = 'a; b; c' RETURN n"
        )


class TestParameterValidation:
    def test_none_becomes_empty_dict(self) -> None:
        assert _validate_parameters(None) == {}

    def test_accepts_json_primitives(self) -> None:
        params = {
            "s": "x",
            "i": 1,
            "f": 1.5,
            "b": True,
            "none": None,
            "lst": [1, "two", None],
            "obj": {"k": "v"},
        }
        assert _validate_parameters(params) == params

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            _validate_parameters([("k", "v")])  # type: ignore[arg-type]

    def test_rejects_unsupported_value_types(self) -> None:
        with pytest.raises(ValueError, match="unsupported type"):
            _validate_parameters({"k": object()})

    def test_rejects_non_string_keys(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            _validate_parameters({1: "v"})  # type: ignore[dict-item]


# --------------------------------------------------------------------------- #
# 2. Graph-type serialisation                                                 #
# --------------------------------------------------------------------------- #


class _FakeNode:
    """Quacks like ``neo4j.graph.Node`` enough for ``_serialize_value``."""

    def __init__(
        self, element_id: str, labels: set[str], properties: dict[str, Any]
    ) -> None:
        self.element_id = element_id
        self.labels = labels
        self._props = properties

    def __iter__(self):  # required for dict(node)
        return iter(self._props)

    def __getitem__(self, key: str) -> Any:
        return self._props[key]

    def keys(self):
        return self._props.keys()


def test_serialize_value_handles_plain_primitives() -> None:
    assert _serialize_value(1) == 1
    assert _serialize_value("x") == "x"
    assert _serialize_value([1, "two", None]) == [1, "two", None]
    assert _serialize_value({"k": [1, {"inner": "v"}]}) == {
        "k": [1, {"inner": "v"}]
    }


def test_serialize_value_handles_temporal_types() -> None:
    from datetime import date, datetime, time, timedelta

    dt = datetime(2026, 5, 29, 3, 59, 27, tzinfo=UTC)
    assert _serialize_value(dt) == "2026-05-29T03:59:27+00:00"
    assert _serialize_value(date(2026, 5, 29)) == "2026-05-29"
    assert _serialize_value(time(3, 59, 27)) == "03:59:27"
    assert _serialize_value(timedelta(seconds=12.5)) == 12.5


def test_serialize_value_handles_bytes() -> None:
    assert _serialize_value(b"\x00\xff") == "00ff"


def test_serialize_value_unpacks_node_objects() -> None:
    fake = _FakeNode(
        element_id="4:abc:1",
        labels={"Symbol", "Function"},
        properties={"fqn": "pkg.f", "kind": "function"},
    )
    # Patch the ``Node`` symbol that ``_serialize_value`` does its
    # isinstance() check against so our fake counts as a Node.
    with patch.object(ec_mod, "Node", _FakeNode):
        out = _serialize_value(fake)
    assert out == {
        "_type": "node",
        "element_id": "4:abc:1",
        "labels": ["Function", "Symbol"],
        "properties": {"fqn": "pkg.f", "kind": "function"},
    }


# --------------------------------------------------------------------------- #
# 3. End-to-end with a fake Neo4j session                                     #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeRecord:
    data: dict[str, Any]

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


@dataclass
class _FakeSummary:
    query_type: str = "r"
    result_available_after: int = 1
    result_consumed_after: int = 2


@dataclass
class _FakeResult:
    records: list[_FakeRecord]
    summary: _FakeSummary
    _consumed: bool = False

    def __iter__(self):
        return iter(self.records)

    def consume(self) -> _FakeSummary:
        self._consumed = True
        return self.summary

    def keys(self):
        return self.records[0].keys() if self.records else []


@dataclass
class _FakeSession:
    """Captures every run + returns scripted results."""

    queries: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    # Map ``"explain" | "run"`` -> the next FakeResult to return.
    next_explain: _FakeResult | None = None
    next_run: _FakeResult | None = None

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def run(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> _FakeResult:
        self.queries.append((query, parameters or {}))
        if query.startswith("EXPLAIN "):
            assert self.next_explain is not None, "EXPLAIN not scripted"
            return self.next_explain
        # The execute_read path runs the real query through ``tx.run``,
        # which our fake session also handles.
        assert self.next_run is not None, "run not scripted"
        return self.next_run

    def begin_transaction(self, *, timeout: float) -> _FakeTransaction:
        # Production code now opens an explicit transaction with a hard
        # timeout. Our fake transaction just delegates ``run`` back to the
        # session so a single recorded list of queries covers both phases.
        assert timeout > 0
        return _FakeTransaction(self)


@dataclass
class _FakeTransaction:
    """Minimal Transaction stand-in returned from ``begin_transaction``."""

    session: _FakeSession
    committed: bool = False

    def __enter__(self) -> _FakeTransaction:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def run(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> _FakeResult:
        return self.session.run(query, parameters=parameters)

    def commit(self) -> None:
        self.committed = True


class _FakeNeo4jContext:
    """Stand-in for ``code_spider.mcp.context.AppContext``."""

    def __init__(self, session: _FakeSession) -> None:
        self.neo4j = self  # both client & session live on the same object
        self._session = session

    def __enter__(self) -> _FakeSession:
        return self._session.__enter__()

    def __exit__(self, *exc: Any) -> None:
        self._session.__exit__(*exc)


@pytest.fixture
def fake_session_ctx():
    """Patch out ``get_context`` and ``read_session`` for an isolated tool run."""
    session = _FakeSession()
    ctx = _FakeNeo4jContext(session)

    with (
        patch.object(ec_mod, "get_context", return_value=ctx),
        patch.object(ec_mod, "read_session", lambda _client: session),
    ):
        yield session


class TestExecuteCypher:
    def test_happy_path_returns_serialised_rows(
        self, fake_session_ctx: _FakeSession
    ) -> None:
        fake_session_ctx.next_explain = _FakeResult(
            records=[], summary=_FakeSummary(query_type="r")
        )
        fake_session_ctx.next_run = _FakeResult(
            records=[
                _FakeRecord({"fqn": "pkg.a", "n": 1}),
                _FakeRecord({"fqn": "pkg.b", "n": 2}),
            ],
            summary=_FakeSummary(query_type="r"),
        )
        result = execute_cypher(
            workspace_id="demo",
            query="MATCH (s:Symbol) RETURN s.fqn AS fqn, s.n AS n LIMIT 2",
            parameters={"workspace_id": "demo"},
            limit=5,
        )
        assert result["row_count"] == 2
        assert result["truncated"] is False
        assert result["query_type"] == "r"
        assert result["columns"] == ["fqn", "n"]
        assert result["rows"] == [
            {"fqn": "pkg.a", "n": 1},
            {"fqn": "pkg.b", "n": 2},
        ]
        # Both EXPLAIN and the real run happened, in that order, with the
        # same parameters dict (no kwparameter collisions).
        assert len(fake_session_ctx.queries) == 2
        explain_q, explain_p = fake_session_ctx.queries[0]
        run_q, run_p = fake_session_ctx.queries[1]
        assert explain_q.startswith("EXPLAIN ")
        assert "EXPLAIN " not in run_q
        assert explain_p == run_p == {"workspace_id": "demo"}

    def test_truncates_at_limit(self, fake_session_ctx: _FakeSession) -> None:
        fake_session_ctx.next_explain = _FakeResult(
            records=[], summary=_FakeSummary(query_type="r")
        )
        fake_session_ctx.next_run = _FakeResult(
            records=[_FakeRecord({"i": i}) for i in range(10)],
            summary=_FakeSummary(query_type="r"),
        )
        result = execute_cypher(
            workspace_id="demo",
            query="MATCH (n) RETURN n.i AS i",
            limit=3,
        )
        assert result["row_count"] == 3
        assert result["truncated"] is True
        assert result["rows"] == [{"i": 0}, {"i": 1}, {"i": 2}]

    def test_rejects_when_explain_reports_write_query(
        self, fake_session_ctx: _FakeSession
    ) -> None:
        # The static scan can't see CALL targets, but EXPLAIN can. We feed
        # the tool a benign-looking statement and let EXPLAIN flag it.
        fake_session_ctx.next_explain = _FakeResult(
            records=[], summary=_FakeSummary(query_type="rw")
        )
        with pytest.raises(PermissionError, match="read-only"):
            execute_cypher(
                workspace_id="demo",
                query="CALL custom.procedure() YIELD x RETURN x",
            )

    def test_rejects_static_write_keyword_before_hitting_neo4j(
        self, fake_session_ctx: _FakeSession
    ) -> None:
        with pytest.raises(PermissionError, match="CREATE"):
            execute_cypher(
                workspace_id="demo",
                query="CREATE (n:Foo) RETURN n",
            )
        # Neo4j was never contacted.
        assert fake_session_ctx.queries == []

    def test_rejects_empty_query(self, fake_session_ctx: _FakeSession) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            execute_cypher(workspace_id="demo", query="   ")

    def test_rejects_oversized_query(
        self, fake_session_ctx: _FakeSession
    ) -> None:
        big = "MATCH (n) RETURN n " + "/* " + "x" * MAX_QUERY_CHARS + " */"
        with pytest.raises(ValueError, match="exceeds"):
            execute_cypher(workspace_id="demo", query=big)

    @pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1])
    def test_rejects_out_of_range_limit(
        self, fake_session_ctx: _FakeSession, limit: int
    ) -> None:
        with pytest.raises(ValueError, match="between 1 and"):
            execute_cypher(
                workspace_id="demo",
                query="MATCH (n) RETURN n",
                limit=limit,
            )

    def test_rejects_multi_statement(
        self, fake_session_ctx: _FakeSession
    ) -> None:
        with pytest.raises(ValueError, match="single Cypher statement"):
            execute_cypher(
                workspace_id="demo",
                query="MATCH (n) RETURN n; MATCH (m) RETURN m",
            )

    def test_uses_default_limit_when_omitted(
        self, fake_session_ctx: _FakeSession
    ) -> None:
        # Just sanity-check that DEFAULT_LIMIT is wired through.
        fake_session_ctx.next_explain = _FakeResult(
            records=[], summary=_FakeSummary(query_type="r")
        )
        fake_session_ctx.next_run = _FakeResult(
            records=[_FakeRecord({"i": i}) for i in range(DEFAULT_LIMIT + 5)],
            summary=_FakeSummary(query_type="r"),
        )
        result = execute_cypher(
            workspace_id="demo", query="MATCH (n) RETURN n.i AS i"
        )
        assert result["row_count"] == DEFAULT_LIMIT
        assert result["truncated"] is True

    def test_rejects_invalid_workspace_id(
        self, fake_session_ctx: _FakeSession
    ) -> None:
        with pytest.raises(ValueError, match="invalid workspace_id"):
            execute_cypher(
                workspace_id="../etc/passwd",
                query="MATCH (n) RETURN n",
            )
