"""Indexing progress reporters.

Two flavours, picked automatically based on whether stderr is a TTY:

* :class:`RichProgressReporter` — renders a live :mod:`rich.progress` bar
  with one overall workspace task plus per-repo sub-tasks. Bar shows the
  total chunks completed / overall ETA / per-second rate, plus a moving
  per-repo bar that updates as sub-batches complete in parallel.
* :class:`LogProgressReporter` — fallback for non-TTY environments (CI
  logs, the MCP server running under a coding agent, ``nohup`` redirects).
  Emits structured :mod:`structlog` lines every ~5 % so users still see
  forward motion in the journal.

The reporter is a context manager: entering starts the bar; exiting closes
it cleanly even on exception. Inside the context, the indexer calls

    progress.start_repo(name, total=N)
    progress.advance_repo(n)        # ... possibly many times
    progress.finish_repo()

per repo. The reporter takes care of routing those calls to either the
rich bar or the log emitter.

Why a class hierarchy instead of just rich-with-Console-redirect? Rich's
``Console(force_terminal=False)`` still emits ANSI escape sequences in
many CI environments; the log reporter is a clean stdout-only fallback
that plays well with structured logging pipelines.
"""

from __future__ import annotations

import sys
import time
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from code_spider.logging_setup import get_logger

_log = get_logger(__name__)


class _ProgressReporter(Protocol):
    """Common surface for progress reporters.

    Both implementations are also context managers; the indexer always
    drives them as ``with _embed_progress(...) as progress:``.
    """

    def start_repo(self, repo_name: str, *, total: int) -> None: ...
    def advance_repo(self, n: int) -> None: ...
    def finish_repo(self) -> None: ...


class RichProgressReporter(AbstractContextManager["RichProgressReporter"]):
    """Live two-row progress: workspace total + current repo.

    Designed for interactive terminals. Falls back to the log reporter
    automatically when stderr isn't a TTY (see :func:`embed_progress`).
    """

    def __init__(self, *, total_chunks: int, workspace_id: str) -> None:
        self._total = total_chunks
        self._workspace = workspace_id
        self._console = Console(stderr=True)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[dim]({task.fields[rate]} chunks/s)"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta"),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
            refresh_per_second=4,
        )
        self._ws_task: TaskID | None = None
        self._repo_task: TaskID | None = None
        self._repo_started_at: float = 0.0
        self._ws_started_at: float = 0.0

    def __enter__(self) -> RichProgressReporter:
        self._progress.start()
        self._ws_started_at = time.perf_counter()
        self._ws_task = self._progress.add_task(
            f"workspace {self._workspace}",
            total=self._total,
            rate="0.0",
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._repo_task is not None:
            self._progress.remove_task(self._repo_task)
            self._repo_task = None
        self._progress.stop()

    def start_repo(self, repo_name: str, *, total: int) -> None:
        if self._repo_task is not None:
            self._progress.remove_task(self._repo_task)
        self._repo_started_at = time.perf_counter()
        self._repo_task = self._progress.add_task(
            f"  repo {repo_name}",
            total=total,
            rate="0.0",
        )

    def advance_repo(self, n: int) -> None:
        # Update per-repo bar.
        if self._repo_task is not None:
            elapsed = max(1e-6, time.perf_counter() - self._repo_started_at)
            done = self._progress.tasks[self._task_index(self._repo_task)].completed + n
            self._progress.update(
                self._repo_task,
                advance=n,
                rate=f"{done / elapsed:.1f}",
            )
        # Update workspace bar.
        if self._ws_task is not None:
            elapsed = max(1e-6, time.perf_counter() - self._ws_started_at)
            done = self._progress.tasks[self._task_index(self._ws_task)].completed + n
            self._progress.update(
                self._ws_task,
                advance=n,
                rate=f"{done / elapsed:.1f}",
            )

    def finish_repo(self) -> None:
        if self._repo_task is not None:
            self._progress.remove_task(self._repo_task)
            self._repo_task = None

    def _task_index(self, task_id: TaskID) -> int:
        # ``Progress.tasks`` is a list; find the matching task. Tiny N (≤2),
        # so a linear scan is fine and avoids relying on internal API.
        for i, t in enumerate(self._progress.tasks):
            if t.id == task_id:
                return i
        raise KeyError(task_id)


class LogProgressReporter(AbstractContextManager["LogProgressReporter"]):
    """Structured-log fallback used when stderr is not a TTY.

    Emits a log line every ~5 % of the per-repo total so log scrapers still
    see forward motion. Cheap; never blocks; safe to call from threads.
    """

    _PROGRESS_STEP_RATIO = 0.05  # log every 5% (or every chunk if total<20)

    def __init__(self, *, total_chunks: int, workspace_id: str) -> None:
        self._workspace = workspace_id
        self._total = total_chunks
        self._repo_name: str | None = None
        self._repo_total = 0
        self._repo_done = 0
        self._repo_next_log_at = 0
        self._repo_started_at = 0.0
        self._ws_started_at = 0.0
        self._ws_done = 0

    def __enter__(self) -> LogProgressReporter:
        self._ws_started_at = time.perf_counter()
        _log.info(
            "embed progress: workspace starting",
            workspace=self._workspace,
            total_chunks=self._total,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed = time.perf_counter() - self._ws_started_at
        rate = self._ws_done / elapsed if elapsed > 0 else 0.0
        _log.info(
            "embed progress: workspace finished",
            workspace=self._workspace,
            done=self._ws_done,
            total=self._total,
            elapsed_s=round(elapsed, 2),
            rate_per_s=round(rate, 1),
        )

    def start_repo(self, repo_name: str, *, total: int) -> None:
        self._repo_name = repo_name
        self._repo_total = total
        self._repo_done = 0
        self._repo_started_at = time.perf_counter()
        step = max(1, int(total * self._PROGRESS_STEP_RATIO))
        self._repo_next_log_at = step
        _log.info(
            "embed progress: repo starting",
            workspace=self._workspace,
            repo=repo_name,
            chunks=total,
        )

    def advance_repo(self, n: int) -> None:
        self._repo_done += n
        self._ws_done += n
        if self._repo_done >= self._repo_next_log_at:
            elapsed = max(1e-6, time.perf_counter() - self._repo_started_at)
            rate = self._repo_done / elapsed
            pct = (self._repo_done / self._repo_total * 100) if self._repo_total else 100.0
            _log.info(
                "embed progress",
                workspace=self._workspace,
                repo=self._repo_name,
                done=self._repo_done,
                total=self._repo_total,
                pct=round(pct, 1),
                rate_per_s=round(rate, 1),
            )
            step = max(1, int(self._repo_total * self._PROGRESS_STEP_RATIO))
            self._repo_next_log_at += step

    def finish_repo(self) -> None:
        if self._repo_name is None:
            return
        elapsed = time.perf_counter() - self._repo_started_at
        rate = self._repo_done / elapsed if elapsed > 0 else 0.0
        _log.info(
            "embed progress: repo finished",
            workspace=self._workspace,
            repo=self._repo_name,
            done=self._repo_done,
            total=self._repo_total,
            elapsed_s=round(elapsed, 2),
            rate_per_s=round(rate, 1),
        )
        self._repo_name = None


def embed_progress(
    *, total_chunks: int, workspace_id: str
) -> AbstractContextManager[_ProgressReporter]:
    """Pick the right reporter based on whether stderr is a TTY.

    Callable separately from the indexer so other long-running stages can
    reuse the same UX in future without re-implementing TTY detection.
    """
    if sys.stderr.isatty():
        return RichProgressReporter(total_chunks=total_chunks, workspace_id=workspace_id)
    return LogProgressReporter(total_chunks=total_chunks, workspace_id=workspace_id)
