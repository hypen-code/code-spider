"""Materialise a repo on disk at a specific commit SHA.

Two modes:
    1. **Local path** (``RepoConfig.path``) — used in dev. The path is taken
       as-is and the *current* commit SHA is read via ``git rev-parse HEAD``.
       If the path is not a git repo, a synthetic SHA derived from the manifest
       + repo name + mtime is used so the rest of the pipeline still functions.
    2. **Remote URL** (``RepoConfig.url``) — used in CI. We clone (``--depth=1``
       --filter=blob:none) into ``<checkout_root>/<workspace>/<repo>@<sha>/``.

Checkouts are keyed by SHA so concurrent indexer runs never collide.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from git import GitCommandError, InvalidGitRepositoryError, Repo

from code_spider.logging_setup import get_logger
from code_spider.workspace.manifest import RepoConfig

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CheckoutResult:
    """A materialised repository ready to be parsed."""

    repo_name: str
    root: Path
    commit_sha: str
    is_local: bool


def ensure_checkout(
    *,
    workspace_id: str,
    repo: RepoConfig,
    checkout_root: Path,
    requested_sha: str | None = None,
) -> CheckoutResult:
    """Return a CheckoutResult, performing a clone/fetch only when needed."""
    if repo.path is not None:
        return _resolve_local(workspace_id=workspace_id, repo=repo)
    if repo.url is None:  # pragma: no cover - validated by Pydantic
        raise ValueError(f"repo '{repo.name}' has neither path nor url")
    return _resolve_remote(
        workspace_id=workspace_id,
        repo=repo,
        checkout_root=checkout_root,
        requested_sha=requested_sha,
    )


# --------------------------------------------------------------------- local


def _resolve_local(workspace_id: str, repo: RepoConfig) -> CheckoutResult:
    assert repo.path is not None
    root = Path(repo.path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"workspace '{workspace_id}' repo '{repo.name}' path does not exist: {root}"
        )

    sha: str
    try:
        sha = Repo(root).head.commit.hexsha
    except (InvalidGitRepositoryError, ValueError):
        sha = _synthetic_sha(workspace_id, repo.name, root)
        _log.warning(
            "local repo is not a git working copy; using synthetic commit sha",
            repo=repo.name,
            sha=sha,
        )

    _log.info("resolved local checkout", repo=repo.name, root=str(root), sha=sha)
    return CheckoutResult(repo_name=repo.name, root=root, commit_sha=sha, is_local=True)


def _synthetic_sha(workspace_id: str, repo_name: str, root: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(workspace_id.encode())
    h.update(b"\x00")
    h.update(repo_name.encode())
    h.update(b"\x00")
    h.update(str(int(root.stat().st_mtime)).encode())
    return h.hexdigest()


# -------------------------------------------------------------------- remote


def _resolve_remote(
    *,
    workspace_id: str,
    repo: RepoConfig,
    checkout_root: Path,
    requested_sha: str | None,
) -> CheckoutResult:
    assert repo.url is not None
    sha = requested_sha or _ls_remote_head(repo.url, repo.branch)
    target = checkout_root / workspace_id / f"{repo.name}@{sha}"
    if target.exists() and (target / ".git").is_dir():
        _log.info("checkout already present", repo=repo.name, sha=sha, root=str(target))
        return CheckoutResult(repo_name=repo.name, root=target, commit_sha=sha, is_local=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)

    _log.info("cloning repo", repo=repo.name, url=repo.url, sha=sha, root=str(target))
    try:
        Repo.clone_from(
            url=repo.url,
            to_path=str(target),
            multi_options=[
                "--filter=blob:none",
                "--no-tags",
                f"--branch={repo.branch}",
            ],
        )
        Repo(target).git.checkout(sha)
    except GitCommandError as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"git clone failed for {repo.url}@{sha}: {exc}") from exc

    return CheckoutResult(repo_name=repo.name, root=target, commit_sha=sha, is_local=False)


def _ls_remote_head(url: str, branch: str) -> str:
    """Resolve the branch tip SHA without cloning."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", url, f"refs/heads/{branch}"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-remote failed for {url}#{branch}: {result.stderr.strip()}"
        )
    line = result.stdout.splitlines()[0] if result.stdout else ""
    sha = line.split("\t", 1)[0].strip()
    if not sha:
        raise RuntimeError(f"could not resolve {url}#{branch} to a SHA")
    return sha
