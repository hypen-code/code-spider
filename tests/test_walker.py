"""Walker + inline-chunking regression tests.

The walker used to populate a workspace-wide ``_SOURCE_CACHE`` and a later
``_chunk_workspace`` stage re-read every file. That design held every
repo's raw bytes in RAM through the resolver + flow-matcher stages and was
the dominant memory cost on a 4 GiB box. The walker now:

* honours ``Settings.max_file_bytes`` and skips oversize files before reading;
* runs the chunker inline so the source bytes are dropped immediately;
* attaches chunks to the returned :class:`FileRecord` so no second pass is needed.

These tests pin that contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_spider.checkout.git import CheckoutResult
from code_spider.indexer import _iter_source_files, _walk_repo
from code_spider.workspace.manifest import RepoConfig


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A minimal Python repo with one normal file + one oversize file."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "calc.py").write_text(
        '"""calc."""\n\n'
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "def sub(a: int, b: int) -> int:\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def checkout(tiny_repo: Path) -> CheckoutResult:
    return CheckoutResult(
        repo_name="fixture",
        root=tiny_repo,
        commit_sha="0" * 40,
        is_local=True,
    )


def _repo() -> RepoConfig:
    return RepoConfig(name="fixture", path="/unused", branch="main", languages=["python"])


# --------------------------------------------------------------------------- #
# Inline chunking — chunks must be attached to FileRecord at walk time        #
# --------------------------------------------------------------------------- #


def test_walker_chunks_inline(checkout: CheckoutResult) -> None:
    files = list(
        _walk_repo(
            workspace_id="ws",
            checkout=checkout,
            repo=_repo(),
            max_file_bytes=1_048_576,
        )
    )
    # Both files yielded; calc.py has two top-level functions ⇒ ≥2 chunks.
    by_name = {f.repo_relative_path: f for f in files}
    assert "pkg/calc.py" in by_name
    assert len(by_name["pkg/calc.py"].chunks) >= 2
    # Empty file produces zero chunks but is still yielded for graph hygiene.
    assert by_name["pkg/__init__.py"].chunks == []


# --------------------------------------------------------------------------- #
# File-size cap — oversize files are skipped before being read                #
# --------------------------------------------------------------------------- #


def test_walker_skips_oversize_files(checkout: CheckoutResult, tiny_repo: Path) -> None:
    # Drop a 5 KiB "minified" file alongside the sources.
    huge = tiny_repo / "pkg" / "huge.py"
    huge.write_text("x = " + repr("a" * 5_000) + "\n", encoding="utf-8")
    assert huge.stat().st_size > 4_096

    files = list(
        _walk_repo(
            workspace_id="ws",
            checkout=checkout,
            repo=_repo(),
            max_file_bytes=4_096,  # 4 KiB cap → huge.py skipped, others kept
        )
    )
    paths = {f.repo_relative_path for f in files}
    assert "pkg/huge.py" not in paths, "oversize file must be skipped"
    assert "pkg/calc.py" in paths, "normal-size file must still be parsed"


def test_walker_cap_zero_disables_skipping(checkout: CheckoutResult, tiny_repo: Path) -> None:
    """``max_file_bytes=0`` disables the cap (sentinel for opt-out)."""
    huge = tiny_repo / "pkg" / "huge.py"
    huge.write_text("y = 1\n" * 10_000, encoding="utf-8")
    files = list(
        _walk_repo(
            workspace_id="ws",
            checkout=checkout,
            repo=_repo(),
            max_file_bytes=0,
        )
    )
    assert "pkg/huge.py" in {f.repo_relative_path for f in files}


# --------------------------------------------------------------------------- #
# Source bytes are dropped — no module-level cache anymore                    #
# --------------------------------------------------------------------------- #


def test_no_global_source_cache_attribute() -> None:
    """The walker must not re-introduce a module-level source cache.

    A workspace-wide cache was the dominant memory cost on 4 GiB boxes —
    one 10 MiB minified bundle in the workspace blew the budget. If anyone
    re-adds a ``_SOURCE_CACHE`` to the indexer module, this test fails so
    the regression is caught at PR time.
    """
    from code_spider import indexer

    assert not hasattr(indexer, "_SOURCE_CACHE"), (
        "indexer._SOURCE_CACHE was removed for memory safety; do not re-add it. "
        "Pass `source` directly to the chunker inside `_walk_repo` instead."
    )


# --------------------------------------------------------------------------- #
# Ignored directory pruning still works                                       #
# --------------------------------------------------------------------------- #


def test_walker_skips_ignored_dirs(tmp_path: Path) -> None:
    """Files inside ``.venv`` / ``node_modules`` etc. are pruned by the iterator."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "vendored.py").write_text("y = 2\n", encoding="utf-8")

    found = {p.relative_to(root).as_posix() for p in _iter_source_files(root)}
    assert "src/main.py" in found
    assert all(not p.startswith(".venv/") for p in found)
