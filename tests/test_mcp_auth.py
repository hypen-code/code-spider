"""MCP auth + audit tests (no Neo4j required)."""

from __future__ import annotations

import time

import pytest

from code_spider.mcp import auth
from code_spider.mcp.auth import (
    assert_safe_identifier,
    assert_safe_workspace_id,
    audited,
)


def test_workspace_id_validator_accepts_valid_ids() -> None:
    for ok in ("demo", "payments-platform", "svc_a", "ws1"):
        assert assert_safe_workspace_id(ok) == ok


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "../etc/passwd",
        "MIXEDcase",
        "spaces here",
        "-leading-dash",
        "ws/with/slash",
    ],
)
def test_workspace_id_validator_rejects_unsafe_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        assert_safe_workspace_id(bad)


def test_identifier_validator_accepts_common_fqn_shapes() -> None:
    for ok in (
        "pkg.module.Class.method",
        "GET",
        "/users/{}",
        "@nestjs/common.Get",
        "http://localhost:7687",
    ):
        assert assert_safe_identifier(ok) == ok


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "drop\ntable",
        "weird;injection",
        "<script>",
        "with whitespace",
    ],
)
def test_identifier_validator_rejects_unsafe(bad: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        assert_safe_identifier(bad)


def test_audited_decorator_passes_value_through() -> None:
    @audited("noop")
    def my_tool(x: int) -> int:
        return x * 2

    assert my_tool(3) == 6


def test_audited_decorator_re_raises_errors() -> None:
    @audited("failing")
    def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        boom()


def test_audited_decorator_times_out_slow_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "_resolve_tool_timeout_s", lambda: 0.05)

    @audited("slow")
    def slow_tool() -> int:
        time.sleep(5)
        return 1

    with pytest.raises(TimeoutError, match="slow timed out after 0.05s"):
        slow_tool()


def test_audited_decorator_disables_timeout_when_non_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "_resolve_tool_timeout_s", lambda: 0.0)

    @audited("unbounded")
    def quick_tool() -> int:
        return 42

    assert quick_tool() == 42
