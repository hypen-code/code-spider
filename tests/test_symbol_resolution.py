"""Unit tests for the private symbol resolver shared by call/impact tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from code_spider.mcp.tools._symbol_resolution import (
    _EXACT_QUERY,
    _FUZZY_QUERY,
    _MIN_SUFFIX_CHARS,
    _SUFFIX_QUERY,
    clamp_candidates,
    lucene_escape,
    resolve,
    resolved_to_payload,
)

# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #


class TestLuceneEscape:
    @pytest.mark.parametrize(
        "raw",
        ["foo", "user.save", "pkg.module.UserService", "save_user_123"],
    )
    def test_non_meta_characters_pass_through_unchanged(self, raw: str) -> None:
        # Dots, underscores and digits are not Lucene meta characters so
        # they must survive escaping verbatim.
        assert lucene_escape(raw) == raw

    @pytest.mark.parametrize(
        "raw",
        ['"', "+", "-", "&", "|", "!", "(", ")", "{", "}", "[", "]", "^",
         "~", "*", "?", ":", "\\", "/"],
    )
    def test_escapes_every_meta_character(self, raw: str) -> None:
        out = lucene_escape(raw)
        assert out.startswith("\\")
        assert out.endswith(raw)
        assert len(out) == 2

    def test_escapes_in_context(self) -> None:
        # Realistic LLM-supplied query containing a colon (Lucene meta).
        assert lucene_escape("UserService:save") == r"UserService\:save"


class TestClampCandidates:
    def test_clamps_to_minimum(self) -> None:
        assert clamp_candidates(0) == 1
        assert clamp_candidates(-5) == 1

    def test_clamps_to_maximum(self) -> None:
        # Hard cap baked into the module.
        assert clamp_candidates(10_000) == 50

    def test_pass_through_in_range(self) -> None:
        assert clamp_candidates(10) == 10
        assert clamp_candidates(1) == 1
        assert clamp_candidates(50) == 50


# --------------------------------------------------------------------------- #
# Fake Neo4j session for mode runners.                                        #
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
    """Routes ``session.run`` based on the Cypher template it sees."""

    by_query: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    seen: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def run(
        self, query: str, parameters: dict[str, Any] | None = None, **kw: Any
    ) -> _Result:
        params = dict(parameters or {})
        params.update(kw)
        self.seen.append((query, params))
        # Match by identity-ish: we keyed responses on the exact template
        # the resolver uses (imported as constants above).
        return _Result(rows=list(self.by_query.get(query, [])))


def _make_row(fqn: str, *, score: float | None = None, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fqn": fqn,
        "repo": extra.get("repo", "demo-repo"),
        "file_path": extra.get("file_path", f"{fqn.replace('.', '/')}.py"),
        "start_line": extra.get("start_line", 1),
        "end_line": extra.get("end_line", 10),
        "kind": extra.get("kind", "function"),
    }
    if score is not None:
        row["score"] = score
    return row


# --------------------------------------------------------------------------- #
# Mode behaviour                                                              #
# --------------------------------------------------------------------------- #


class TestExactMode:
    def test_returns_single_exact_match(self) -> None:
        session = _FakeSession(
            by_query={_EXACT_QUERY: [_make_row("pkg.module.fn")]},
        )
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="pkg.module.fn",
            mode="exact",
            max_candidates=10,
        )
        assert mode == "exact"
        assert len(rows) == 1
        assert rows[0].fqn == "pkg.module.fn"
        assert rows[0].mode == "exact"
        assert rows[0].score == 1.0

    def test_none_when_no_match(self) -> None:
        session = _FakeSession(by_query={_EXACT_QUERY: []})
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="nope",
            mode="exact",
            max_candidates=10,
        )
        assert rows == []
        assert mode is None

    def test_passes_limit_to_session(self) -> None:
        session = _FakeSession(by_query={_EXACT_QUERY: []})
        resolve(
            session,
            workspace_id="demo",
            fqn="x",
            mode="exact",
            max_candidates=7,
        )
        _, params = session.seen[0]
        assert params["limit"] == 7
        assert params["fqn"] == "x"
        assert params["workspace_id"] == "demo"


class TestSuffixMode:
    def test_returns_results_sorted_shortest_first(self) -> None:
        session = _FakeSession(
            by_query={
                _SUFFIX_QUERY: [
                    _make_row("pkg.User.save"),
                    _make_row("other.pkg.User.save"),
                ]
            }
        )
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="User.save",
            mode="suffix",
            max_candidates=10,
        )
        assert mode == "suffix"
        assert [r.fqn for r in rows] == [
            "pkg.User.save",
            "other.pkg.User.save",
        ]
        # All rows are tagged as suffix-mode hits.
        assert all(r.mode == "suffix" for r in rows)

    def test_rejects_too_short_tail(self) -> None:
        session = _FakeSession(by_query={_SUFFIX_QUERY: [_make_row("x")]})
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="x" * (_MIN_SUFFIX_CHARS - 1),
            mode="suffix",
            max_candidates=10,
        )
        assert rows == []
        assert mode is None
        # The resolver short-circuits without even touching the session.
        assert session.seen == []


class TestFuzzyMode:
    def test_wraps_query_in_phrase_and_escapes_meta(self) -> None:
        session = _FakeSession(
            by_query={_FUZZY_QUERY: [_make_row("svc.UserService.save", score=2.3)]}
        )
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="UserService:save",  # contains Lucene meta
            mode="fuzzy",
            max_candidates=10,
        )
        assert mode == "fuzzy"
        assert rows[0].score == pytest.approx(2.3)
        assert rows[0].mode == "fuzzy"
        # Verify the Lucene query was wrapped in quotes + colon escaped.
        _, params = session.seen[0]
        assert params["q"] == r'"UserService\:save"'

    def test_blank_query_returns_no_rows_without_hitting_session(self) -> None:
        session = _FakeSession(by_query={_FUZZY_QUERY: [_make_row("x", score=1.0)]})
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="   ",
            mode="fuzzy",
            max_candidates=10,
        )
        assert rows == []
        assert mode is None
        assert session.seen == []


class TestAutoMode:
    def test_returns_exact_when_available(self) -> None:
        session = _FakeSession(
            by_query={
                _EXACT_QUERY: [_make_row("a.b.c")],
                # Should not be consulted because exact already matched.
                _SUFFIX_QUERY: [_make_row("zzz.a.b.c")],
                _FUZZY_QUERY: [_make_row("yyy.a.b.c", score=9.9)],
            }
        )
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="a.b.c",
            mode="auto",
            max_candidates=10,
        )
        assert mode == "exact"
        assert [r.fqn for r in rows] == ["a.b.c"]
        # Only one Cypher run; the auto cascade stopped at exact.
        assert len(session.seen) == 1

    def test_falls_back_to_suffix_when_exact_empty(self) -> None:
        session = _FakeSession(
            by_query={
                _EXACT_QUERY: [],
                _SUFFIX_QUERY: [_make_row("pkg.User.save")],
                _FUZZY_QUERY: [_make_row("noisy", score=0.1)],
            }
        )
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="User.save",
            mode="auto",
            max_candidates=10,
        )
        assert mode == "suffix"
        assert [r.fqn for r in rows] == ["pkg.User.save"]
        # exact + suffix were probed; fuzzy was skipped.
        assert len(session.seen) == 2

    def test_falls_through_to_fuzzy(self) -> None:
        session = _FakeSession(
            by_query={
                _EXACT_QUERY: [],
                _SUFFIX_QUERY: [],
                _FUZZY_QUERY: [_make_row("svc.User.save_user", score=1.5)],
            }
        )
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="save user",
            mode="auto",
            max_candidates=10,
        )
        assert mode == "fuzzy"
        assert rows[0].score == pytest.approx(1.5)
        assert len(session.seen) == 3

    def test_returns_none_when_every_mode_empty(self) -> None:
        session = _FakeSession(
            by_query={_EXACT_QUERY: [], _SUFFIX_QUERY: [], _FUZZY_QUERY: []}
        )
        rows, mode = resolve(
            session,
            workspace_id="demo",
            fqn="ghost",
            mode="auto",
            max_candidates=10,
        )
        assert rows == []
        assert mode is None


# --------------------------------------------------------------------------- #
# Payload shape                                                               #
# --------------------------------------------------------------------------- #


def test_resolved_to_payload_round_trips_dataclass() -> None:
    session = _FakeSession(
        by_query={_EXACT_QUERY: [_make_row("pkg.fn", start_line=12, end_line=34)]}
    )
    rows, _ = resolve(
        session,
        workspace_id="demo",
        fqn="pkg.fn",
        mode="exact",
        max_candidates=1,
    )
    payload = resolved_to_payload(rows[0])
    assert payload == {
        "fqn": "pkg.fn",
        "repo": "demo-repo",
        "file_path": "pkg/fn.py",
        "start_line": 12,
        "end_line": 34,
        "kind": "function",
        "score": 1.0,
        "match_mode": "exact",
    }
