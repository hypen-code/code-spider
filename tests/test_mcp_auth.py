"""MCP auth + audit tests (no Neo4j required)."""

from __future__ import annotations

import pytest

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
