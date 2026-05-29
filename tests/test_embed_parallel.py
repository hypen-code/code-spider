"""Concurrency contract for the parallel-sub-batch embed path.

``_embed_one_repo`` partitions a repo's chunk texts into ``batch_size``
slices and submits them to a :class:`ThreadPoolExecutor` with ``workers``
threads. Because embedding is I/O-bound (provider socket reads), threading
is the right primitive and the GIL is released during the network wait.

These tests pin three properties:

1. **Order preservation** — futures complete out of order; results must
   still land at the correct chunk positions.
2. **Concurrency actually happens** — when ``workers >= 2``, two slices
   are in flight at the same time. We assert this with a barrier-like
   stub that blocks until N callers arrive.
3. **Fault isolation** — a failing sub-batch must not corrupt the
   neighbouring slices; the affected chunks get no embedding and the rest
   still receive theirs.
"""

from __future__ import annotations

import threading
import time

from code_spider.embedding.provider import EmbeddingProvider
from code_spider.indexer import _embed_one_repo
from code_spider.progress import LogProgressReporter
from code_spider.symbols.model import (
    Chunk,
    FileRecord,
    ParseResult,
    Span,
    WorkspaceParseBundle,
)


def _make_chunk(idx: int, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"c{idx:03d}",
        file_path="f.py",
        span=Span(start_line=1, start_col=0, end_line=1, end_col=1),
        text=text if text is not None else f"chunk-{idx}",
    )


def _bundle(num_chunks: int) -> WorkspaceParseBundle:
    chunks = [_make_chunk(i) for i in range(num_chunks)]
    return WorkspaceParseBundle(
        workspace_id="ws",
        workspace_name="ws",
        manifest_sha="0" * 64,
        repos=[
            ParseResult(
                workspace_id="ws",
                repo_name="alpha",
                commit_sha="0" * 40,
                files=[
                    FileRecord(
                        repo_relative_path="f.py",
                        lang="python",
                        hash_blake3="dead",
                        size_bytes=10,
                        line_count=1,
                        module=None,
                        chunks=chunks,
                    )
                ],
            )
        ],
    )


def _silent_progress() -> LogProgressReporter:
    """A reporter that is entered but produces no output during the test."""
    return LogProgressReporter(total_chunks=0, workspace_id="ws")


# --------------------------------------------------------------------------- #
# 1. Order preservation                                                       #
# --------------------------------------------------------------------------- #


class _DeterministicProvider(EmbeddingProvider):
    """Each input embeds to a vector keyed by its text — easy to verify order."""

    name = "deterministic"
    dim = 1

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(int(t.split("-")[1]))] for t in texts]


def test_parallel_path_preserves_chunk_order() -> None:
    bundle = _bundle(num_chunks=20)
    provider = _DeterministicProvider()
    progress = _silent_progress()
    with progress:
        applied = _embed_one_repo(
            bundle=bundle,
            r_idx=0,
            provider=provider,
            workers=4,
            batch_size=3,  # 20/3 → 7 slices, exercises the boundary case
            progress=progress,
        )
    assert applied == 20
    # Each chunk's embedding must equal its own index — proves results
    # landed in the original slot regardless of future completion order.
    for i, c in enumerate(bundle.repos[0].files[0].chunks):
        assert c.embedding == (float(i),), f"chunk {i} got embedding {c.embedding!r}"


# --------------------------------------------------------------------------- #
# 2. Real concurrency under workers >= 2                                      #
# --------------------------------------------------------------------------- #


class _BarrierProvider(EmbeddingProvider):
    """Blocks each call until ``parties`` callers have arrived.

    If the executor is actually parallel (workers >= parties), the barrier
    releases and every call completes. If the indexer accidentally
    serialised the calls (workers=1, or a missing thread-pool), the test
    deadlocks at ``barrier.wait()`` and pytest's per-test timeout fires.
    """

    name = "barrier"
    dim = 1

    def __init__(self, *, parties: int, timeout_s: float = 5.0) -> None:
        self._barrier = threading.Barrier(parties)
        self._timeout = timeout_s
        self._max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self._in_flight += 1
            self._max_in_flight = max(self._max_in_flight, self._in_flight)
        try:
            # Wait for `parties` concurrent callers — proves the executor
            # really did dispatch slices in parallel.
            self._barrier.wait(timeout=self._timeout)
        finally:
            with self._lock:
                self._in_flight -= 1
        return [[float(t[-1] == "9")] for t in texts]


def test_parallel_path_runs_calls_concurrently() -> None:
    """``workers=3`` ⇒ at least 3 calls are in flight at the same time."""
    bundle = _bundle(num_chunks=9)  # 9/3 = 3 slices, each will block on the barrier
    provider = _BarrierProvider(parties=3, timeout_s=5.0)
    progress = _silent_progress()
    with progress:
        applied = _embed_one_repo(
            bundle=bundle,
            r_idx=0,
            provider=provider,
            workers=3,
            batch_size=3,
            progress=progress,
        )
    assert applied == 9
    assert provider._max_in_flight == 3, (
        f"expected 3 concurrent calls, observed {provider._max_in_flight}"
    )


# --------------------------------------------------------------------------- #
# 3. Fault isolation between sub-batches                                      #
# --------------------------------------------------------------------------- #


class _OneBatchFailsProvider(EmbeddingProvider):
    """Fail any sub-batch that contains the chunk whose text ends in '7'."""

    name = "one-fail"
    dim = 1

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any(t.endswith("-7") for t in texts):
            raise RuntimeError("simulated transient provider error")
        return [[1.0] for _ in texts]


def test_one_failing_subbatch_does_not_block_others() -> None:
    bundle = _bundle(num_chunks=12)
    provider = _OneBatchFailsProvider()
    progress = _silent_progress()
    with progress:
        applied = _embed_one_repo(
            bundle=bundle,
            r_idx=0,
            provider=provider,
            workers=3,
            batch_size=3,  # slices of 3 → the slice [6,7,8] fails, others succeed
            progress=progress,
        )
    # Only the failing slice (3 chunks) lacks embeddings; the other 9 land.
    assert applied == 9

    chunks = bundle.repos[0].files[0].chunks
    # Chunk 7's slice [6..9) all failed → no embedding for those three.
    for i in (6, 7, 8):
        assert chunks[i].embedding == (), f"chunk {i} should have no embedding"
    # All other chunks did get an embedding.
    for i in (0, 1, 2, 3, 4, 5, 9, 10, 11):
        assert chunks[i].embedding == (1.0,), f"chunk {i} should be embedded"


# --------------------------------------------------------------------------- #
# 4. workers=1 fast path is exercised (no executor overhead)                  #
# --------------------------------------------------------------------------- #


class _CountingSerialProvider(EmbeddingProvider):
    """Records the call order so the test can confirm strict serialisation."""

    name = "serial"
    dim = 1

    def __init__(self) -> None:
        self.call_order: list[int] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Tiny sleep makes any accidental thread interleaving visible.
        time.sleep(0.005)
        self.call_order.append(int(texts[0].split("-")[1]))
        return [[1.0] for _ in texts]


def test_workers_eq_one_runs_serially_in_input_order() -> None:
    bundle = _bundle(num_chunks=6)
    provider = _CountingSerialProvider()
    progress = _silent_progress()
    with progress:
        _embed_one_repo(
            bundle=bundle,
            r_idx=0,
            provider=provider,
            workers=1,
            batch_size=2,
            progress=progress,
        )
    # Single-thread fast path → slices processed strictly in order.
    assert provider.call_order == [0, 2, 4]
