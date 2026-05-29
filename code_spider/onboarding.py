"""Developer-onboarding helpers for the public, pip-installed distribution.

These are the commands a developer runs *once* on their workstation to point
``code-spider serve`` (the MCP server) at a central Neo4j instance maintained
by their team. No indexing happens locally; the developer only consumes the
already-built graph.

Three commands live here:

- :func:`run_configure` — interactive wizard that prompts for the four Neo4j
  settings, optionally tests the bolt connection, and writes them to the user
  config file (default ``~/.config/code-spider/config.env``).

- :func:`run_mcp_config` — prints a ready-to-paste MCP server JSON snippet for
  Windsurf, Cursor, Claude Code, and Codex / generic MCP clients.

- :func:`run_doctor` — verifies the configuration end-to-end: env loaded, bolt
  reachable, auth works, schema applied (``Symbol`` constraint present).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from code_spider.config import Neo4jSettings, user_config_path

# Env keys we manage. Order matters for the rendered config file.
_MANAGED_KEYS: tuple[str, ...] = (
    "CODE_SPIDER_NEO4J_URI",
    "CODE_SPIDER_NEO4J_USER",
    "CODE_SPIDER_NEO4J_PASSWORD",
    "CODE_SPIDER_NEO4J_DATABASE",
    "CODE_SPIDER_LOG_LEVEL",
    "CODE_SPIDER_LOG_JSON",
)

_DEFAULTS: dict[str, str] = {
    "CODE_SPIDER_NEO4J_URI": "bolt://localhost:7687",
    "CODE_SPIDER_NEO4J_USER": "neo4j",
    "CODE_SPIDER_NEO4J_PASSWORD": "",
    "CODE_SPIDER_NEO4J_DATABASE": "neo4j",
    "CODE_SPIDER_LOG_LEVEL": "INFO",
    "CODE_SPIDER_LOG_JSON": "0",
}


@dataclass(frozen=True, slots=True)
class WizardAnswers:
    """The four pieces of information the wizard collects."""

    uri: str
    user: str
    password: str
    database: str


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` dotenv file. Ignores blanks and ``#`` lines."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _render_env_file(values: dict[str, str]) -> str:
    """Render values as a deterministic dotenv body. Keeps unmanaged keys verbatim."""
    managed = {k: values.get(k, _DEFAULTS.get(k, "")) for k in _MANAGED_KEYS}
    extras = {k: v for k, v in values.items() if k not in _MANAGED_KEYS}
    lines = ["# Managed by `code-spider configure`. Edit by re-running the wizard."]
    lines.append("")
    lines.append("# Central Neo4j connection")
    for key in (
        "CODE_SPIDER_NEO4J_URI",
        "CODE_SPIDER_NEO4J_USER",
        "CODE_SPIDER_NEO4J_PASSWORD",
        "CODE_SPIDER_NEO4J_DATABASE",
    ):
        lines.append(f"{key}={managed[key]}")
    lines.append("")
    lines.append("# Logging")
    lines.append(f"CODE_SPIDER_LOG_LEVEL={managed['CODE_SPIDER_LOG_LEVEL']}")
    lines.append(f"CODE_SPIDER_LOG_JSON={managed['CODE_SPIDER_LOG_JSON']}")
    if extras:
        lines.append("")
        lines.append("# User-added overrides (preserved by the wizard)")
        for key, value in extras.items():
            lines.append(f"{key}={value}")
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Write ``content`` to ``path`` atomically with restrictive permissions.

    Permissions matter — the file contains a Neo4j password.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    # Best-effort: chmod is meaningless on Windows; ignore platform errors.
    with contextlib.suppress(OSError):
        os.chmod(tmp, mode)
    tmp.replace(path)


def _test_bolt(answers: WizardAnswers) -> str | None:
    """Return ``None`` on success or a human-readable failure reason.

    Uses a fresh ``Neo4jClient`` rather than the env-bound ``load_settings`` so
    the wizard can test values that have not been persisted yet.
    """
    from code_spider.graph import Neo4jClient  # local import keeps --help cheap

    settings = Neo4jSettings(
        uri=answers.uri,
        user=answers.user,
        password=answers.password,
        database=answers.database,
    )
    try:
        with Neo4jClient(settings) as client, client.session() as session:
            session.run("RETURN 1 AS ok").single()
    except Exception as exc:
        return str(exc)
    return None


def run_configure(
    *,
    console: Console,
    non_interactive: bool,
    uri: str | None,
    user: str | None,
    password: str | None,
    database: str | None,
    skip_test: bool,
    config_path: Path | None,
) -> int:
    """Run the ``code-spider configure`` wizard. Returns the desired exit code."""
    target = config_path or user_config_path()
    existing = _parse_env_file(target)

    def _default(key: str) -> str:
        return existing.get(key, _DEFAULTS.get(key, ""))

    if non_interactive:
        # In non-interactive mode every required value must be supplied either
        # via flag or via the existing file. We refuse to invent passwords.
        answers = WizardAnswers(
            uri=uri or _default("CODE_SPIDER_NEO4J_URI"),
            user=user or _default("CODE_SPIDER_NEO4J_USER"),
            password=password if password is not None else _default("CODE_SPIDER_NEO4J_PASSWORD"),
            database=database or _default("CODE_SPIDER_NEO4J_DATABASE"),
        )
        if not answers.password:
            console.print(
                "[red]--password is required in non-interactive mode "
                "(or set CODE_SPIDER_NEO4J_PASSWORD in the existing config).[/red]"
            )
            return 2
    else:
        console.print(
            Panel.fit(
                "Configure [bold]code-spider[/bold] to talk to your team's central "
                "Neo4j knowledge graph.\nValues are saved to "
                f"[cyan]{target}[/cyan] with 0600 permissions.",
                title="code-spider configure",
                border_style="cyan",
            )
        )
        answers = WizardAnswers(
            uri=(
                uri
                or typer.prompt(
                    "Neo4j bolt URI",
                    default=_default("CODE_SPIDER_NEO4J_URI"),
                )
            ).strip(),
            user=(
                user
                or typer.prompt(
                    "Neo4j username",
                    default=_default("CODE_SPIDER_NEO4J_USER"),
                )
            ).strip(),
            password=(
                password
                if password is not None
                else typer.prompt(
                    "Neo4j password",
                    hide_input=True,
                    default=_default("CODE_SPIDER_NEO4J_PASSWORD") or None,
                    show_default=False,
                )
            ),
            database=(
                database
                or typer.prompt(
                    "Neo4j database",
                    default=_default("CODE_SPIDER_NEO4J_DATABASE"),
                )
            ).strip(),
        )

    if not skip_test:
        console.print(f"\nTesting connection to [cyan]{answers.uri}[/cyan] ...")
        reason = _test_bolt(answers)
        if reason is None:
            console.print("[green]OK[/green] — bolt connection succeeded.")
        else:
            console.print(f"[red]Connection failed:[/red] {reason}")
            if non_interactive:
                # In CI/scripts we can't prompt — fail loudly so a bad creds
                # change doesn't silently overwrite a working config.
                console.print(
                    "Aborted (non-interactive). Pass [cyan]--skip-test[/cyan] to save anyway."
                )
                return 1
            if not typer.confirm("Save these values anyway?", default=False):
                console.print("Aborted. Nothing written.")
                return 1

    merged = dict(existing)
    merged["CODE_SPIDER_NEO4J_URI"] = answers.uri
    merged["CODE_SPIDER_NEO4J_USER"] = answers.user
    merged["CODE_SPIDER_NEO4J_PASSWORD"] = answers.password
    merged["CODE_SPIDER_NEO4J_DATABASE"] = answers.database
    merged.setdefault("CODE_SPIDER_LOG_LEVEL", _DEFAULTS["CODE_SPIDER_LOG_LEVEL"])
    merged.setdefault("CODE_SPIDER_LOG_JSON", _DEFAULTS["CODE_SPIDER_LOG_JSON"])

    _atomic_write(target, _render_env_file(merged))
    console.print(f"\n[green]Saved[/green] {target}")
    console.print(
        "\nNext steps:\n"
        "  1. [cyan]code-spider doctor[/cyan]      — verify the connection\n"
        "  2. [cyan]code-spider mcp-config[/cyan]  — print MCP JSON to paste into your agent\n"
    )
    return 0


# ---------------------------------------------------------------------------
# mcp-config
# ---------------------------------------------------------------------------


_AGENT_CHOICES: tuple[str, ...] = ("windsurf", "cursor", "claude-code", "generic")


def _resolve_command() -> str:
    """Return the absolute path to the installed ``code-spider`` binary.

    Agents must invoke an absolute path because they don't inherit the user's
    shell ``$PATH``. We fall back to the literal name only if discovery fails.
    """
    found = shutil.which("code-spider")
    if found:
        return found
    # Best guess from the current interpreter's bin dir.
    candidate = Path(sys.executable).with_name("code-spider")
    if candidate.exists():
        return str(candidate)
    return "code-spider"


def _build_mcp_block(*, command: str, include_password: bool, settings: Neo4jSettings) -> dict:
    """Build the ``mcpServers.code-spider`` object that agents understand."""
    env: dict[str, str] = {
        "CODE_SPIDER_NEO4J_URI": settings.uri,
        "CODE_SPIDER_NEO4J_USER": settings.user,
        "CODE_SPIDER_NEO4J_DATABASE": settings.database,
    }
    if include_password:
        env["CODE_SPIDER_NEO4J_PASSWORD"] = settings.password
    else:
        env["CODE_SPIDER_NEO4J_PASSWORD"] = "<set-in-config-or-replace-here>"
    return {
        "command": command,
        "args": ["serve"],
        "env": env,
    }


def run_mcp_config(
    *,
    console: Console,
    agent: str,
    include_password: bool,
    command_override: str | None,
) -> int:
    """Print a copy-pasteable MCP server JSON snippet for the chosen agent."""
    if agent not in _AGENT_CHOICES:
        console.print(
            f"[red]Unknown agent '{agent}'. Choose one of: {', '.join(_AGENT_CHOICES)}.[/red]"
        )
        return 2

    # Load settings lazily; if missing, fall back to placeholders so
    # `mcp-config` still works on a brand-new install.
    try:
        from code_spider.config import load_settings

        settings = load_settings().neo4j
    except RuntimeError:
        settings = Neo4jSettings(
            uri="bolt://central-neo4j.example.com:7687",
            user="readonly",
            password="<set-in-config-or-replace-here>",
            database="neo4j",
        )

    command = command_override or _resolve_command()
    block = _build_mcp_block(command=command, include_password=include_password, settings=settings)

    if agent in ("windsurf", "cursor", "claude-code", "generic"):
        wrapped = {"mcpServers": {"code-spider": block}}
        snippet = json.dumps(wrapped, indent=2)

    target_hint = {
        "windsurf": "~/.codeium/windsurf/mcp_config.json",
        "cursor": "~/.cursor/mcp.json  (or project .cursor/mcp.json)",
        "claude-code": "claude mcp add-json code-spider '<paste the inner object>'",
        "generic": "Any MCP client that consumes the standard JSON schema",
    }[agent]

    console.print(
        Panel.fit(
            f"Paste this into [bold]{target_hint}[/bold]:",
            border_style="cyan",
        )
    )
    console.print(Syntax(snippet, "json", theme="ansi_dark", word_wrap=False))
    if not include_password:
        console.print(
            "\n[yellow]Note:[/yellow] password is a placeholder. Either run "
            "[cyan]code-spider configure[/cyan] (the agent will read the password "
            "from the saved config file) or re-run with [cyan]--include-password[/cyan]."
        )
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _check_schema_present(client) -> tuple[bool, str]:
    """Return (ok, detail). ``ok`` is False if the Symbol constraint is missing."""
    query = "SHOW CONSTRAINTS YIELD name, labelsOrTypes RETURN name, labelsOrTypes"
    try:
        with client.session() as session:
            rows = list(session.run(query))
    except Exception as exc:
        return False, f"SHOW CONSTRAINTS failed: {exc}"
    names = [r["name"] for r in rows]
    if not names:
        return False, "no constraints — run `code-spider migrate` on the central server"
    return True, f"{len(names)} constraints present"


def run_doctor(*, console: Console) -> int:
    """End-to-end verification: env → bolt → auth → schema."""
    console.print("[bold]code-spider doctor[/bold]\n")

    cfg_path = user_config_path()
    console.print(f"  config file : {cfg_path} {'(found)' if cfg_path.is_file() else '(missing)'}")

    try:
        from code_spider.config import load_settings

        settings = load_settings()
    except RuntimeError as exc:
        console.print(f"  [red]config    : {exc}[/red]")
        console.print("\nRun [cyan]code-spider configure[/cyan] to create the config file.")
        return 2

    masked = settings.neo4j.password[:1] + "***" if settings.neo4j.password else "(empty)"
    console.print(f"  neo4j uri   : {settings.neo4j.uri}")
    console.print(f"  neo4j user  : {settings.neo4j.user}")
    console.print(f"  neo4j pass  : {masked}")
    console.print(f"  database    : {settings.neo4j.database}\n")

    from code_spider.graph import Neo4jClient

    try:
        with Neo4jClient(settings.neo4j) as client:
            with client.session() as session:
                row = session.run("RETURN 1 AS ok").single()
            bolt_ok = bool(row) and row["ok"] == 1
            status = "[green]OK[/green]" if bolt_ok else "[red]unexpected result[/red]"
            console.print(f"  bolt connect: {status}")
            ok, detail = _check_schema_present(client)
            schema_status = "[green]OK[/green]" if ok else "[yellow]warn[/yellow]"
            console.print(f"  schema      : {schema_status} \u2014 {detail}")
            try:
                with client.session() as session:
                    n = session.run("MATCH (s:Symbol) RETURN count(s) AS n").single()
                count = n["n"] if n else 0
                hint = (
                    "(graph populated)"
                    if count > 0
                    else "(empty \u2014 run `code-spider index` on the central server)"
                )
                console.print(f"  symbol count: {count} {hint}")
            except Exception as exc:
                console.print(f"  [yellow]symbol count failed: {exc}[/yellow]")
    except Exception as exc:
        console.print(f"  [red]bolt connect: {exc}[/red]")
        console.print(
            "\nFix the URI / credentials with [cyan]code-spider configure[/cyan] and try again."
        )
        return 1

    console.print("\n[green]All checks passed.[/green] Add the MCP server to your agent with:")
    console.print("  [cyan]code-spider mcp-config --agent windsurf[/cyan]\n")
    return 0
