"""Unit tests for the ``code-spider configure | mcp-config | doctor`` commands.

These tests stay hermetic: bolt connection is monkey-patched, the user
config path is redirected to ``tmp_path``, and no Neo4j is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from code_spider import onboarding
from code_spider.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# user_config_path / XDG override
# ---------------------------------------------------------------------------


def test_user_config_path_uses_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "explicit.env"
    monkeypatch.setenv("CODE_SPIDER_CONFIG_FILE", str(target))
    from code_spider.config import user_config_path

    assert user_config_path() == target


def test_user_config_path_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CODE_SPIDER_CONFIG_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from code_spider.config import user_config_path

    assert user_config_path() == tmp_path / "code-spider" / "config.env"


def test_user_config_path_defaults_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CODE_SPIDER_CONFIG_FILE", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from code_spider.config import user_config_path

    assert user_config_path() == tmp_path / ".config" / "code-spider" / "config.env"


# ---------------------------------------------------------------------------
# env file round-trip
# ---------------------------------------------------------------------------


def test_parse_and_render_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "config.env"
    src.write_text(
        "# header\n"
        "CODE_SPIDER_NEO4J_URI=bolt://srv:7687\n"
        "CODE_SPIDER_NEO4J_USER=ro\n"
        "CODE_SPIDER_NEO4J_PASSWORD=p4ss\n"
        "CODE_SPIDER_NEO4J_DATABASE=neo4j\n"
        "CODE_SPIDER_LOG_LEVEL=DEBUG\n"
        "CODE_SPIDER_LOG_JSON=1\n"
        "MY_EXTRA=stay\n",
        encoding="utf-8",
    )
    parsed = onboarding._parse_env_file(src)
    assert parsed["CODE_SPIDER_NEO4J_URI"] == "bolt://srv:7687"
    assert parsed["MY_EXTRA"] == "stay"

    rendered = onboarding._render_env_file(parsed)
    assert "CODE_SPIDER_NEO4J_URI=bolt://srv:7687" in rendered
    assert "CODE_SPIDER_LOG_LEVEL=DEBUG" in rendered
    # Extras must survive a round trip.
    assert "MY_EXTRA=stay" in rendered


def test_render_handles_quoted_values(tmp_path: Path) -> None:
    src = tmp_path / "config.env"
    src.write_text('CODE_SPIDER_NEO4J_PASSWORD="quoted secret"\n', encoding="utf-8")
    parsed = onboarding._parse_env_file(src)
    assert parsed["CODE_SPIDER_NEO4J_PASSWORD"] == "quoted secret"


def test_atomic_write_sets_0600(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "config.env"
    onboarding._atomic_write(target, "X=1\n")
    assert target.read_text() == "X=1\n"
    mode = target.stat().st_mode & 0o777
    # On POSIX we should see 0600; on platforms where chmod is a no-op the
    # bits may differ — only enforce when chmod actually took effect.
    if mode != 0:
        assert mode == 0o600


# ---------------------------------------------------------------------------
# configure (non-interactive)
# ---------------------------------------------------------------------------


def test_configure_non_interactive_writes_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "config.env"
    # Stub the bolt probe so we don't need a live Neo4j.
    monkeypatch.setattr(onboarding, "_test_bolt", lambda _a: None)

    result = runner.invoke(
        app,
        [
            "configure",
            "--non-interactive",
            "--uri",
            "bolt://central:7687",
            "--user",
            "ro",
            "--password",
            "secret",
            "--database",
            "neo4j",
            "--config-path",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    body = target.read_text()
    assert "CODE_SPIDER_NEO4J_URI=bolt://central:7687" in body
    assert "CODE_SPIDER_NEO4J_PASSWORD=secret" in body


def test_configure_non_interactive_rejects_missing_password(tmp_path: Path) -> None:
    target = tmp_path / "config.env"
    result = runner.invoke(
        app,
        [
            "configure",
            "--non-interactive",
            "--uri",
            "bolt://central:7687",
            "--user",
            "ro",
            "--config-path",
            str(target),
        ],
    )
    assert result.exit_code == 2
    assert "password" in result.output.lower()
    assert not target.exists()


def test_configure_bolt_failure_aborts_in_non_interactive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "config.env"
    monkeypatch.setattr(onboarding, "_test_bolt", lambda _a: "auth failed")

    result = runner.invoke(
        app,
        [
            "configure",
            "--non-interactive",
            "--uri",
            "bolt://central:7687",
            "--user",
            "ro",
            "--password",
            "wrong",
            "--config-path",
            str(target),
        ],
    )
    # Non-interactive + failed probe + no `--skip-test` → exit 1 and don't write.
    assert result.exit_code == 1
    assert not target.exists()


def test_configure_skip_test_writes_even_when_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "config.env"
    # Probe must not be called when --skip-test is set; guard it.
    monkeypatch.setattr(
        onboarding,
        "_test_bolt",
        lambda _a: pytest.fail("bolt probe should be skipped"),
    )

    result = runner.invoke(
        app,
        [
            "configure",
            "--non-interactive",
            "--skip-test",
            "--uri",
            "bolt://unreachable:7687",
            "--user",
            "ro",
            "--password",
            "x",
            "--config-path",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    assert target.is_file()


# ---------------------------------------------------------------------------
# mcp-config
# ---------------------------------------------------------------------------


def _extract_json(output: str) -> dict:
    """Pull the first balanced JSON object out of rich-rendered output."""
    start = output.find("{")
    assert start != -1, f"no JSON in output:\n{output}"
    depth = 0
    end = -1
    for i, ch in enumerate(output[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end != -1
    return json.loads(output[start:end])


def test_mcp_config_prints_windsurf_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "config.env"
    target.write_text(
        "CODE_SPIDER_NEO4J_URI=bolt://central:7687\n"
        "CODE_SPIDER_NEO4J_USER=ro\n"
        "CODE_SPIDER_NEO4J_PASSWORD=verysecret\n"
        "CODE_SPIDER_NEO4J_DATABASE=neo4j\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_SPIDER_CONFIG_FILE", str(target))
    # Re-load so the test sees the new file values.
    monkeypatch.setenv("CODE_SPIDER_NEO4J_URI", "bolt://central:7687")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_USER", "ro")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_PASSWORD", "verysecret")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_DATABASE", "neo4j")

    result = runner.invoke(
        app, ["mcp-config", "--agent", "windsurf", "--command", "/opt/bin/code-spider"]
    )
    assert result.exit_code == 0, result.output
    cfg = _extract_json(result.output)
    server = cfg["mcpServers"]["code-spider"]
    assert server["command"] == "/opt/bin/code-spider"
    assert server["args"] == ["serve"]
    assert server["env"]["CODE_SPIDER_NEO4J_URI"] == "bolt://central:7687"
    # Password is a placeholder by default.
    assert server["env"]["CODE_SPIDER_NEO4J_PASSWORD"] != "verysecret"


def test_mcp_config_include_password(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODE_SPIDER_NEO4J_URI", "bolt://x:7687")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_USER", "u")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_PASSWORD", "real-pw")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_DATABASE", "neo4j")

    result = runner.invoke(
        app,
        [
            "mcp-config",
            "--agent",
            "cursor",
            "--include-password",
            "--command",
            "/opt/bin/code-spider",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = _extract_json(result.output)
    assert cfg["mcpServers"]["code-spider"]["env"]["CODE_SPIDER_NEO4J_PASSWORD"] == "real-pw"


def test_mcp_config_rejects_unknown_agent() -> None:
    result = runner.invoke(app, ["mcp-config", "--agent", "atom"])
    assert result.exit_code == 2
    assert "Unknown agent" in result.output


def test_mcp_config_works_without_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Simulate a fresh install: no config file, no env vars. ``load_settings``
    # still returns sane localhost defaults (documented behavior), so mcp-config
    # must succeed and render a usable starting-point JSON block.
    monkeypatch.setenv("CODE_SPIDER_CONFIG_FILE", str(tmp_path / "absent.env"))
    for key in (
        "CODE_SPIDER_NEO4J_URI",
        "CODE_SPIDER_NEO4J_USER",
        "CODE_SPIDER_NEO4J_PASSWORD",
        "CODE_SPIDER_NEO4J_DATABASE",
    ):
        monkeypatch.delenv(key, raising=False)

    result = runner.invoke(app, ["mcp-config", "--agent", "windsurf"])
    assert result.exit_code == 0, result.output
    cfg = _extract_json(result.output)
    env = cfg["mcpServers"]["code-spider"]["env"]
    # Either the localhost default (current behavior) or the example.com
    # placeholder is acceptable — both clearly signal "you must edit me".
    assert env["CODE_SPIDER_NEO4J_URI"] in {
        "bolt://localhost:7687",
        "bolt://central-neo4j.example.com:7687",
    }
    # Password must be a placeholder, never a real value, when --include-password
    # was not passed.
    assert env["CODE_SPIDER_NEO4J_PASSWORD"] != "codespider-dev-password"


# ---------------------------------------------------------------------------
# doctor (no live Neo4j — Neo4jClient is monkey-patched)
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, *, schema_rows: list[dict], symbol_count: int) -> None:
        self._schema_rows = schema_rows
        self._symbol_count = symbol_count

    def run(self, query: str):
        class _Result:
            def __init__(self, payload):
                self._payload = payload

            def single(self):
                return self._payload

            def __iter__(self):
                if isinstance(self._payload, list):
                    return iter(self._payload)
                return iter([self._payload])

        if "RETURN 1" in query:
            return _Result({"ok": 1})
        if "SHOW CONSTRAINTS" in query:
            return _Result(self._schema_rows)
        if "count(s)" in query:
            return _Result({"n": self._symbol_count})
        return _Result({})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _FakeClient:
    def __init__(self, *, schema_rows: list[dict], symbol_count: int) -> None:
        self._schema_rows = schema_rows
        self._symbol_count = symbol_count

    def session(self):
        return _FakeSession(schema_rows=self._schema_rows, symbol_count=self._symbol_count)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_doctor_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_SPIDER_NEO4J_URI", "bolt://x:7687")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_USER", "u")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_PASSWORD", "p")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_DATABASE", "neo4j")

    from code_spider import graph as graph_pkg

    monkeypatch.setattr(
        graph_pkg,
        "Neo4jClient",
        lambda _settings: _FakeClient(
            schema_rows=[{"name": "Symbol_fqn_unique", "labelsOrTypes": ["Symbol"]}],
            symbol_count=42,
        ),
    )

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "bolt connect" in result.output
    assert "All checks passed" in result.output


def test_doctor_reports_bolt_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_SPIDER_NEO4J_URI", "bolt://x:7687")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_USER", "u")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_PASSWORD", "p")
    monkeypatch.setenv("CODE_SPIDER_NEO4J_DATABASE", "neo4j")

    from code_spider import graph as graph_pkg

    def _boom(_settings):
        raise RuntimeError("simulated bolt failure")

    monkeypatch.setattr(graph_pkg, "Neo4jClient", _boom)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "simulated bolt failure" in result.output
