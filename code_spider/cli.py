"""``code-spider`` command-line entry point.

Subcommands:
    configure     Interactive wizard to point the CLI at a central Neo4j.
    doctor        Verify config + bolt connection + schema end-to-end.
    mcp-config    Print a copy-pasteable MCP server JSON snippet.
    migrate       Apply Neo4j constraints + indexes (admin / central server).
    index         Index one or more repos in a workspace (admin / CI).
    serve         Start the MCP server (JSON-RPC over stdio) for an agent.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from code_spider import __version__
from code_spider.config import Settings, load_settings
from code_spider.graph import Neo4jClient, apply_schema
from code_spider.indexer import index_workspace
from code_spider.logging_setup import configure_logging, get_logger
from code_spider.workspace.manifest import load_manifest

app = typer.Typer(
    name="code-spider",
    help="Centralized codebase knowledge graph for AI coding agents.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
_log = get_logger(__name__)


def _settings_or_exit() -> Settings:
    try:
        s = load_settings()
    except RuntimeError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    configure_logging(s.log_level, json_output=s.log_json)
    return s


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"code-spider {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Print version and exit.",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """Code Spider — codebase knowledge graph indexer + MCP server."""


@app.command()
def migrate() -> None:
    """Apply Neo4j schema (constraints, fulltext + vector indexes). Idempotent."""
    settings = _settings_or_exit()
    with Neo4jClient(settings.neo4j) as client:
        client.verify()
        apply_schema(client)
    console.print("[green]Schema applied.[/green]")


@app.command()
def index(
    workspace: str = typer.Option(..., "--workspace", "-w", help="Workspace id from the manifest."),
    repo: str | None = typer.Option(None, "--repo", "-r", help="Index only this single repo."),
    manifest_path: Path | None = typer.Option(
        None, "--manifest", "-m", help="Override manifest path."
    ),
    embed: str = typer.Option(
        "auto",
        "--embed",
        help="Embedding provider: auto | sentence-transformers | hash | none.",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental/--full",
        help="Per-file BLAKE3 diff; only reparse changed files. Default is full refresh.",
    ),
    metrics_port: int | None = typer.Option(
        None,
        "--metrics-port",
        help="Expose Prometheus metrics on this port (disabled if unset).",
    ),
) -> None:
    """Run the indexing pipeline for a workspace."""
    settings = _settings_or_exit()
    manifest_file = manifest_path or settings.manifest_path
    try:
        manifest = load_manifest(manifest_file)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    try:
        manifest.workspace(workspace)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if metrics_port is not None:
        from code_spider.observability import start_metrics_server

        start_metrics_server(metrics_port)

    result = index_workspace(
        manifest=manifest,
        workspace_id=workspace,
        settings=settings,
        only_repo=repo,
        embed_provider=embed,
        incremental=incremental,
    )

    if not result["repos"]:
        console.print("[yellow]No repos indexed (check --repo filter).[/yellow]")
        raise typer.Exit(code=1)

    mode_label = result.get("mode", "full")
    table = Table(
        title=f"Indexed workspace '{workspace}' ({mode_label})",
        show_lines=False,
    )
    table.add_column("repo", style="cyan")
    table.add_column("commit", style="dim")
    table.add_column("files", justify="right")
    table.add_column("symbols", justify="right")
    table.add_column("routes", justify="right")
    table.add_column("kafka", justify="right")
    table.add_column("chunks", justify="right")
    for r in result["repos"]:
        kafka_total = r.get("kafka_producers", 0) + r.get("kafka_consumers", 0)
        table.add_row(
            str(r["repo"]),
            str(r["commit"])[:12],
            str(r.get("files", 0)),
            str(r.get("symbols", 0)),
            str(r.get("routes", 0)),
            str(kafka_total),
            str(r.get("chunks", 0)),
        )
    console.print(table)

    console.print(
        f"[dim]http_flows={result['http_flows']} "
        f"kafka_flows={result['kafka_flows']} "
        f"resolver={result.get('resolver', {})}[/dim]"
    )


@app.command()
def serve(
    embed: str = typer.Option(
        "auto",
        "--embed",
        help="Embedding provider for semantic_code_search: auto | sentence-transformers | hash.",
    ),
) -> None:
    """Start the MCP server (JSON-RPC over stdio)."""
    settings = _settings_or_exit()
    # Lazy import so the heavy `mcp` SDK is only required for `serve`.
    from code_spider.mcp.server import run_stdio

    run_stdio(settings=settings, embed_provider=embed)


@app.command()
def configure(
    uri: str | None = typer.Option(None, "--uri", help="Neo4j bolt URI (skips the prompt)."),
    user: str | None = typer.Option(None, "--user", help="Neo4j username (skips the prompt)."),
    password: str | None = typer.Option(
        None,
        "--password",
        help="Neo4j password (skips the prompt). Required in --non-interactive mode.",
    ),
    database: str | None = typer.Option(
        None, "--database", help="Neo4j database name (skips the prompt)."
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-y",
        help="Do not prompt; use flags + existing config only. Useful in CI / scripts.",
    ),
    skip_test: bool = typer.Option(
        False, "--skip-test", help="Don't probe the bolt connection before saving."
    ),
    config_path: Path | None = typer.Option(
        None, "--config-path", help="Write to this file instead of the default user config."
    ),
) -> None:
    """Interactive wizard: point ``code-spider`` at a central Neo4j and save creds."""
    from code_spider.onboarding import run_configure

    code = run_configure(
        console=console,
        non_interactive=non_interactive,
        uri=uri,
        user=user,
        password=password,
        database=database,
        skip_test=skip_test,
        config_path=config_path,
    )
    if code != 0:
        raise typer.Exit(code=code)


@app.command("mcp-config")
def mcp_config(
    agent: str = typer.Option(
        "windsurf",
        "--agent",
        "-a",
        help="Target agent: windsurf | cursor | claude-code | generic.",
    ),
    include_password: bool = typer.Option(
        False,
        "--include-password",
        help="Embed the saved password into the JSON (default: placeholder).",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        help="Override the `command` field (default: absolute path of installed code-spider).",
    ),
) -> None:
    """Print a copy-pasteable MCP server JSON snippet for the chosen agent."""
    from code_spider.onboarding import run_mcp_config

    code = run_mcp_config(
        console=console,
        agent=agent,
        include_password=include_password,
        command_override=command,
    )
    if code != 0:
        raise typer.Exit(code=code)


@app.command()
def doctor() -> None:
    """Verify env → bolt → auth → schema. Run this after ``configure``."""
    from code_spider.onboarding import run_doctor

    code = run_doctor(console=console)
    if code != 0:
        raise typer.Exit(code=code)


if __name__ == "__main__":  # pragma: no cover
    app()
