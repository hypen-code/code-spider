"""Regression + smoke tests for the fuzzy resolver integration into
``get_call_graph`` and ``get_impact_analysis``.

Both tools are exercised against a fake Neo4j session so the test runs
without a live database. We cover:

* default ``match_mode='exact'`` behaviour preserves the historical response
  shape (``target``/``callers``/``callees`` for call graph;
  ``target``/``upstream_symbols``/``handles_routes``/``http_inbound``/
  ``kafka_inbound`` for impact);
* the new keys (``candidates``, ``match_mode_used``) are present on every
  return path;
* invalid ``match_mode`` is rejected.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import pytest

# Resolve the actual submodules (they are re-exported as functions at the
# parent package level, so ``patch.object`` cannot reach the module
# globals without ``importlib.import_module``).
cg_mod = importlib.import_module("code_spider.mcp.tools.get_call_graph")
ia_mod = importlib.import_module("code_spider.mcp.tools.get_impact_analysis")
sr_mod = importlib.import_module("code_spider.mcp.tools._symbol_resolution")


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _Result:
    rows: list[dict[str, Any]]

    def __iter__(self):
        return iter(self.rows)

    def single(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


@dataclass
class _FakeSession:
    responses: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    seen: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def run(
        self, query: str, parameters: dict[str, Any] | None = None, **kw: Any
    ) -> _Result:
        params = dict(parameters or {})
        params.update(kw)
        self.seen.append((query, params))
        return _Result(rows=list(self.responses.get(query, [])))


def _row(fqn: str, **extra: Any) -> dict[str, Any]:
    return {
        "fqn": fqn,
        "repo": extra.get("repo", "demo-repo"),
        "file_path": extra.get("file_path", f"{fqn.replace('.', '/')}.py"),
        "start_line": extra.get("start_line", 1),
        "end_line": extra.get("end_line", 5),
        "kind": extra.get("kind", "function"),
    }


def _call_graph_traversal_row(fqn: str) -> dict[str, Any]:
    """The shape ``get_call_graph._QUERY`` is expected to return."""
    return {
        "target": {
            "fqn": fqn,
            "repo": "demo-repo",
            "file_path": "x.py",
            "start_line": 1,
            "end_line": 5,
        },
        "callers": [
            {
                "fqn": "caller.fn",
                "repo": "demo-repo",
                "file_path": "c.py",
                "start_line": 1,
                "hops": 1,
                "min_confidence": 0.9,
            }
        ],
        "callees": [],
    }


def _impact_traversal_row(fqn: str) -> dict[str, Any]:
    return {
        "target": {
            "fqn": fqn,
            "repo": "demo-repo",
            "file_path": "x.py",
            "start_line": 1,
            "end_line": 5,
        },
        "upstream_symbols": [],
        "handles_routes": [],
        "http_inbound": [],
        "kafka_inbound": [],
    }


# --------------------------------------------------------------------------- #
# Shared context patch                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture
def patched_session(monkeypatch: pytest.MonkeyPatch):
    """Yield a :class:`_FakeSession` wired into both tool modules."""
    session = _FakeSession()

    class _Ctx:
        neo4j = object()

    def _fake_get_context() -> _Ctx:
        return _Ctx()

    def _fake_read_session(_client: object) -> _FakeSession:
        return session

    monkeypatch.setattr(cg_mod, "get_context", _fake_get_context)
    monkeypatch.setattr(cg_mod, "read_session", _fake_read_session)
    monkeypatch.setattr(ia_mod, "get_context", _fake_get_context)
    monkeypatch.setattr(ia_mod, "read_session", _fake_read_session)
    return session


# --------------------------------------------------------------------------- #
# get_call_graph                                                              #
# --------------------------------------------------------------------------- #


class TestGetCallGraph:
    def test_exact_mode_preserves_legacy_response_keys(
        self, patched_session: _FakeSession
    ) -> None:
        patched_session.responses = {
            sr_mod._EXACT_QUERY: [_row("pkg.fn")],
            cg_mod._QUERY % {"depth": 2}: [_call_graph_traversal_row("pkg.fn")],
        }
        result = cg_mod.get_call_graph(
            workspace_id="demo",
            function_fqn="pkg.fn",
        )
        # Backward-compat keys.
        assert result["target"]["fqn"] == "pkg.fn"
        assert len(result["callers"]) == 1
        assert result["callees"] == []
        # New keys.
        assert result["match_mode_used"] == "exact"
        assert [c["fqn"] for c in result["candidates"]] == ["pkg.fn"]

    def test_returns_empty_response_when_target_missing(
        self, patched_session: _FakeSession
    ) -> None:
        patched_session.responses = {sr_mod._EXACT_QUERY: []}
        result = cg_mod.get_call_graph(
            workspace_id="demo",
            function_fqn="ghost",
        )
        assert result == {
            "target": None,
            "callers": [],
            "callees": [],
            "candidates": [],
            "match_mode_used": None,
        }

    def test_auto_mode_falls_through_to_fuzzy(
        self, patched_session: _FakeSession
    ) -> None:
        primary = "svc.UserService.save_user"
        patched_session.responses = {
            sr_mod._EXACT_QUERY: [],
            sr_mod._SUFFIX_QUERY: [],
            sr_mod._FUZZY_QUERY: [{**_row(primary), "score": 1.7}],
            cg_mod._QUERY % {"depth": 2}: [_call_graph_traversal_row(primary)],
        }
        result = cg_mod.get_call_graph(
            workspace_id="demo",
            function_fqn="save user",
            match_mode="auto",
        )
        assert result["match_mode_used"] == "fuzzy"
        assert result["target"]["fqn"] == primary
        assert result["candidates"][0]["score"] == pytest.approx(1.7)

    def test_rejects_invalid_match_mode(self) -> None:
        with pytest.raises(ValueError, match="invalid match_mode"):
            cg_mod.get_call_graph(
                workspace_id="demo",
                function_fqn="pkg.fn",
                match_mode="bogus",  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------------- #
# get_impact_analysis                                                         #
# --------------------------------------------------------------------------- #


class TestGetImpactAnalysis:
    def test_exact_mode_preserves_legacy_response_keys(
        self, patched_session: _FakeSession
    ) -> None:
        patched_session.responses = {
            sr_mod._EXACT_QUERY: [_row("pkg.fn")],
            ia_mod._QUERY: [_impact_traversal_row("pkg.fn")],
        }
        result = ia_mod.get_impact_analysis(
            workspace_id="demo",
            symbol_fqn="pkg.fn",
        )
        # All legacy keys present.
        for k in (
            "target",
            "upstream_symbols",
            "handles_routes",
            "http_inbound",
            "kafka_inbound",
        ):
            assert k in result
        assert result["target"]["fqn"] == "pkg.fn"
        # New keys present.
        assert result["match_mode_used"] == "exact"
        assert len(result["candidates"]) == 1

    def test_suffix_mode_picks_top_candidate_for_traversal(
        self, patched_session: _FakeSession
    ) -> None:
        # The suffix runner sorts by fqn length ASC so the FIRST candidate
        # is the smallest; we assert the tool uses it for the traversal.
        patched_session.responses = {
            sr_mod._SUFFIX_QUERY: [
                _row("pkg.User.save"),
                _row("other.pkg.User.save"),
            ],
            ia_mod._QUERY: [_impact_traversal_row("pkg.User.save")],
        }
        result = ia_mod.get_impact_analysis(
            workspace_id="demo",
            symbol_fqn="User.save",
            match_mode="suffix",
        )
        assert result["match_mode_used"] == "suffix"
        assert result["target"]["fqn"] == "pkg.User.save"
        # Both candidates surfaced for disambiguation.
        assert [c["fqn"] for c in result["candidates"]] == [
            "pkg.User.save",
            "other.pkg.User.save",
        ]

    def test_returns_empty_response_when_nothing_resolves(
        self, patched_session: _FakeSession
    ) -> None:
        patched_session.responses = {
            sr_mod._EXACT_QUERY: [],
            sr_mod._SUFFIX_QUERY: [],
            sr_mod._FUZZY_QUERY: [],
        }
        result = ia_mod.get_impact_analysis(
            workspace_id="demo",
            symbol_fqn="ghost",
            match_mode="auto",
        )
        assert result["target"] is None
        assert result["candidates"] == []
        assert result["match_mode_used"] is None
