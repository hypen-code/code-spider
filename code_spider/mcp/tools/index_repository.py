"""``index_repository`` — trigger an indexing run from an MCP client.

In CI workflows the indexer is invoked directly by the pipeline; this tool
provides an interactive equivalent for ad-hoc agent-driven reindexes.
"""

from __future__ import annotations

from typing import Any

from code_spider.indexer import index_workspace
from code_spider.mcp.auth import (
    assert_safe_identifier,
    assert_safe_workspace_id,
    audited,
)
from code_spider.mcp.context import get_context
from code_spider.workspace.manifest import load_manifest


@audited("index_repository", timeout_setting="index_timeout_s")
def index_repository(
    workspace_id: str,
    repo: str | None = None,
    embed: str = "auto",
) -> dict[str, Any]:
    """Trigger a full indexing run for ``workspace_id`` (optionally one repo).

    ⚠️ HUMAN-APPROVAL REQUIRED — this is a heavy, mutating, minutes-long
    operation (clones repos, re-parses every file, re-embeds chunks, writes to
    the shared graph). Strict invocation rules:

    1. ALWAYS ask the user for explicit confirmation BEFORE calling this tool,
       stating the ``workspace_id`` (and ``repo`` if scoped) you intend to
       reindex.
    2. If the user declines or does not clearly approve, DO NOT call this tool —
       gracefully drop the request and continue without erroring.
    3. NEVER initiate this tool under auto-approval / autonomous / unattended
       modes. It must only run after a deliberate, in-the-loop human "yes".

    Returns the same stats dict as ``code-spider index``.
    """
    assert_safe_workspace_id(workspace_id)
    if repo is not None:
        assert_safe_identifier(repo, max_len=128)
    ctx = get_context()
    manifest = load_manifest(ctx.settings.manifest_path)
    manifest.workspace(workspace_id)  # validates that it exists
    return index_workspace(
        manifest=manifest,
        workspace_id=workspace_id,
        settings=ctx.settings,
        only_repo=repo,
        embed_provider=embed,
    )
