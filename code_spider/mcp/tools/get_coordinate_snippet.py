"""``get_coordinate_snippet`` — fetch raw text by ``(repo, file, start, end)``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from code_spider.mcp.auth import (
    assert_safe_identifier,
    assert_safe_workspace_id,
    audited,
    read_session,
)
from code_spider.mcp.context import get_context

_REPO_QUERY = """
MATCH (r:Repository {workspace_id: $workspace_id, name: $repo})
RETURN r.path AS path, r.url AS url, r.last_indexed_sha AS sha
"""


@audited("get_coordinate_snippet")
def get_coordinate_snippet(
    workspace_id: str,
    repo: str,
    file_path: str,
    start_line: int,
    end_line: int,
    commit_sha: str | None = None,
) -> dict[str, Any]:
    """Return the raw text between ``start_line`` and ``end_line`` (inclusive).

    The MCP server reads from the indexer's shared checkout directory:
    ``<checkout_root>/<workspace_id>/<repo>@<sha>/<file_path>`` for remote
    repos, or the original ``RepoConfig.path`` for local repos.
    """
    assert_safe_workspace_id(workspace_id)
    assert_safe_identifier(repo, max_len=128)
    assert_safe_identifier(file_path, max_len=1024)
    if start_line < 1 or end_line < start_line:
        raise ValueError("invalid line range")

    ctx = get_context()
    with read_session(ctx.neo4j) as session:
        row = session.run(
            _REPO_QUERY, workspace_id=workspace_id, repo=repo
        ).single()
    if row is None:
        raise FileNotFoundError(f"repo not indexed: {workspace_id}/{repo}")

    sha = commit_sha or row["sha"]
    repo_root = _resolve_repo_root(
        settings_checkout_root=ctx.settings.checkout_root,
        workspace_id=workspace_id,
        repo=repo,
        sha=sha,
        local_path=row["path"],
    )
    abs_path = (repo_root / file_path).resolve()
    if not _is_safe_subpath(repo_root, abs_path):
        raise PermissionError(f"path escapes repo root: {file_path}")
    if not abs_path.is_file():
        raise FileNotFoundError(f"file not found in checkout: {file_path}")

    with abs_path.open("r", encoding="utf-8", errors="replace") as handle:
        all_lines = handle.read().splitlines()
    end_line = min(end_line, len(all_lines))
    snippet = "\n".join(all_lines[start_line - 1 : end_line])
    return {
        "repo": repo,
        "file_path": file_path,
        "commit_sha": sha,
        "start_line": start_line,
        "end_line": end_line,
        "text": snippet,
        "line_count": end_line - start_line + 1,
    }


def _resolve_repo_root(
    *,
    settings_checkout_root: Path,
    workspace_id: str,
    repo: str,
    sha: str | None,
    local_path: str | None,
) -> Path:
    if local_path:
        return Path(local_path).expanduser().resolve()
    if not sha:
        raise FileNotFoundError(f"no commit sha available for {workspace_id}/{repo}")
    return (
        settings_checkout_root / workspace_id / f"{repo}@{sha}"
    ).resolve()


def _is_safe_subpath(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
