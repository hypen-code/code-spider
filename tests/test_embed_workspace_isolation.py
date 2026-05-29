"""Per-repo isolation of the embed stage in :func:`_embed_workspace`.

Regression test for the original failure mode:

    A single 10 MB chunk in one repo triggered a 422 from the embedding
    provider, the exception propagated out of ``embed_batch``, and the entire
    workspace embed stage aborted — losing the embeddings of every other repo
    that had already been processed (and the chunks of the failing repo
    itself, even the well-formed ones).

The new contract:

    * Each repo is embedded as its own independent batch.
    * If a repo's ``embed_batch`` raises, the failure is logged and the loop
      proceeds to the next repo.
    * Repos that succeeded keep their embeddings; the final workspace write
      still persists all of them.
"""

from __future__ import annotations

from code_spider.embedding.provider import EmbeddingProvider
from code_spider.indexer import _embed_workspace
from code_spider.symbols.model import (
    Chunk,
    FileRecord,
    ParseResult,
    Span,
    WorkspaceParseBundle,
)


def _make_chunk(chunk_id: str, text: str = "abc") -> Chunk:
    """Minimal Chunk for tests (defaults are deliberately uninteresting)."""
    return Chunk(
        chunk_id=chunk_id,
        file_path="f.py",
        span=Span(start_line=1, start_col=0, end_line=1, end_col=1),
        text=text,
    )


def _make_file(name: str, chunks: list[Chunk]) -> FileRecord:
    return FileRecord(
        repo_relative_path=name,
        lang="python",
        hash_blake3="deadbeef",
        size_bytes=10,
        line_count=1,
        module=None,
        chunks=chunks,
    )


def _make_repo(name: str, files: list[FileRecord]) -> ParseResult:
    return ParseResult(
        workspace_id="ws",
        repo_name=name,
        commit_sha="0" * 40,
        files=files,
    )


def _make_bundle(repos: list[ParseResult]) -> WorkspaceParseBundle:
    return WorkspaceParseBundle(
        workspace_id="ws",
        workspace_name="ws",
        manifest_sha="0" * 64,
        repos=list(repos),
    )


class _OneRepoFailsProvider(EmbeddingProvider):
    """Fake provider whose ``embed_batch`` raises only for one targeted repo.

    Identifies the offending repo by a sentinel text the test puts in its
    chunks; the alternative — counting calls — would couple the test to
    iteration order.
    """

    name = "fake-fail"
    dim = 4

    def __init__(self, *, poison_text: str) -> None:
        self.poison_text = poison_text
        self.calls: list[int] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        if any(t == self.poison_text for t in texts):
            raise RuntimeError("simulated provider failure (HTTP 422): input length exceeds cap")
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_embed_failure_in_one_repo_does_not_block_other_repos() -> None:
    """If repo B's embed raises, repos A and C must still receive vectors."""
    repo_a = _make_repo(
        "alpha",
        [_make_file("a.py", [_make_chunk("a1", "good"), _make_chunk("a2", "good")])],
    )
    repo_b = _make_repo(
        "beta",
        [_make_file("b.py", [_make_chunk("b1", "POISON")])],
    )
    repo_c = _make_repo(
        "gamma",
        [_make_file("c.py", [_make_chunk("c1", "good")])],
    )
    bundle = _make_bundle([repo_a, repo_b, repo_c])

    provider = _OneRepoFailsProvider(poison_text="POISON")
    _embed_workspace(bundle=bundle, provider=provider, expected_dim=4)

    # Provider was invoked once per repo — proving the loop kept going past
    # the failing batch.
    assert provider.calls == [2, 1, 1]

    # Surviving repos must have real embeddings.
    assert bundle.repos[0].files[0].chunks[0].embedding == (1.0, 0.0, 0.0, 0.0)
    assert bundle.repos[0].files[0].chunks[1].embedding == (1.0, 0.0, 0.0, 0.0)
    assert bundle.repos[2].files[0].chunks[0].embedding == (1.0, 0.0, 0.0, 0.0)

    # The failing repo's chunks remain un-embedded (empty tuple). They are
    # still structurally indexed downstream; only vector search loses
    # coverage for them.
    assert bundle.repos[1].files[0].chunks[0].embedding == ()


def test_empty_repos_are_skipped_without_calling_provider() -> None:
    """Provider must not be called for a repo with zero chunks."""
    repo_empty = _make_repo("empty", [_make_file("e.py", [])])
    repo_real = _make_repo("real", [_make_file("r.py", [_make_chunk("r1")])])
    bundle = _make_bundle([repo_empty, repo_real])

    provider = _OneRepoFailsProvider(poison_text="never matches")
    _embed_workspace(bundle=bundle, provider=provider, expected_dim=4)

    # Only the non-empty repo triggered an SDK call.
    assert provider.calls == [1]
    assert bundle.repos[1].files[0].chunks[0].embedding == (1.0, 0.0, 0.0, 0.0)


def test_dim_mismatch_raises_before_any_provider_call() -> None:
    """Operator-error fast-fail must still happen — silent zero-vectoring a
    dim mismatch would corrupt the Neo4j vector index."""
    import pytest

    bundle = _make_bundle([_make_repo("alpha", [_make_file("a.py", [_make_chunk("a1")])])])
    provider = _OneRepoFailsProvider(poison_text="x")
    # provider.dim == 4, expected_dim=8 — must blow up immediately.
    with pytest.raises(RuntimeError, match="dim=4"):
        _embed_workspace(bundle=bundle, provider=provider, expected_dim=8)
    assert provider.calls == [], "provider must not be invoked on dim mismatch"
