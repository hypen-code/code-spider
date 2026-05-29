"""Progress reporter selection + log-fallback contract.

In a TTY we render a live :mod:`rich.progress` bar; everywhere else (CI,
``code-spider serve`` under an MCP agent, ``nohup``-redirected runs) we
fall back to structured log lines so users still see forward motion.

These tests pin:

* The factory picks ``LogProgressReporter`` when stderr is not a TTY.
* The log reporter emits periodic ``embed progress`` lines (so the user
  isn't staring at silence during a long run).
* Both reporters are usable as context managers and tolerate zero-chunk
  inputs without crashing.
"""

from __future__ import annotations

import io
import sys

import pytest

from code_spider.progress import (
    LogProgressReporter,
    RichProgressReporter,
    embed_progress,
)

# --------------------------------------------------------------------------- #
# Factory: TTY vs non-TTY                                                     #
# --------------------------------------------------------------------------- #


def test_factory_returns_log_reporter_when_stderr_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # io.StringIO is not a TTY → log reporter must be selected.
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    reporter = embed_progress(total_chunks=10, workspace_id="ws")
    assert isinstance(reporter, LogProgressReporter)


def test_factory_returns_rich_reporter_when_stderr_is_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    reporter = embed_progress(total_chunks=10, workspace_id="ws")
    assert isinstance(reporter, RichProgressReporter)


# --------------------------------------------------------------------------- #
# Log reporter: emits progress lines                                          #
# --------------------------------------------------------------------------- #


def test_log_reporter_full_lifecycle_does_not_raise() -> None:
    """Smoke test: enter → start_repo → advance → finish → exit."""
    reporter = LogProgressReporter(total_chunks=100, workspace_id="ws")
    with reporter:
        reporter.start_repo("alpha", total=50)
        for _ in range(50):
            reporter.advance_repo(1)
        reporter.finish_repo()
        reporter.start_repo("beta", total=50)
        for _ in range(50):
            reporter.advance_repo(1)
        reporter.finish_repo()


def test_log_reporter_advance_without_start_is_safe() -> None:
    """``advance_repo`` before ``start_repo`` is a no-op, not a crash.

    Defensive: simplifies the indexer because it never has to special-case
    an empty repo before starting the per-repo task.
    """
    reporter = LogProgressReporter(total_chunks=0, workspace_id="ws")
    with reporter:
        reporter.advance_repo(1)
        reporter.finish_repo()


# --------------------------------------------------------------------------- #
# Rich reporter: also works (smoke test, no assertions on render content)     #
# --------------------------------------------------------------------------- #


def test_rich_reporter_full_lifecycle_does_not_raise() -> None:
    reporter = RichProgressReporter(total_chunks=10, workspace_id="ws")
    with reporter:
        reporter.start_repo("alpha", total=10)
        reporter.advance_repo(3)
        reporter.advance_repo(7)
        reporter.finish_repo()
