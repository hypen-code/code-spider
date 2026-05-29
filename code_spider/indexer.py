"""End-to-end indexing pipeline (Phase 1).

Stages, in order:

    1. **Resolve workspace** — load manifest, validate.
    2. **Checkout**          — per repo, materialise commit-SHA-keyed directory.
    3. **Walk + parse**      — per file, dispatch to language adapter →
                               :class:`FileRecord` with hash + symbols + imports
                               + calls + routes + http_clients + kafka_*.
    4. **Resolve calls**     — workspace-wide :class:`SymbolIndex` +
                               :class:`ImportMap` → ``ResolvedCall`` records.
    5. **Match HTTP_FLOW**   — cross-service client→route edges.
    6. **Match KAFKA_FLOW**  — cross-service producer→consumer edges via topic.
    7. **Chunk**             — AST-aware splits, populated into ``FileRecord.chunks``.
    8. **Embed** (optional)  — :class:`EmbeddingProvider` populates chunk vectors.
    9. **Write**             — full workspace bundle persisted via :class:`GraphWriter`.

Public entry point: :func:`index_workspace`. The CLI (:mod:`code_spider.cli`)
formats the returned stats dict for human consumption.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import blake3

from code_spider.checkout import CheckoutResult, ensure_checkout
from code_spider.chunker import chunk_file
from code_spider.config import Settings
from code_spider.embedding.provider import EmbeddingProvider, get_embedding_provider
from code_spider.graph import GraphWriter, Neo4jClient
from code_spider.incremental import FileDiff, compute_diff, fetch_existing_hashes
from code_spider.logging_setup import get_logger
from code_spider.messaging.kafka_matcher import match_kafka_flows
from code_spider.observability import METRICS, stage_timer
from code_spider.parser import get_adapter
from code_spider.progress import _ProgressReporter
from code_spider.progress import embed_progress as _embed_progress
from code_spider.resolver import resolve_workspace
from code_spider.routes.matcher import match_http_flows
from code_spider.symbols.model import (
    FileRecord,
    ParseResult,
    WorkspaceParseBundle,
)
from code_spider.workspace.manifest import (
    Manifest,
    RepoConfig,
    WorkspaceConfig,
    manifest_sha,
)

_log = get_logger(__name__)


# Directories that are never useful for code intelligence.
_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        ".next",
        ".nuxt",
        ".cache",
        "coverage",
        "neo4j_data",
        "neo4j_logs",
        "neo4j_plugins",
        "neo4j_import",
    }
)


def index_workspace(
    *,
    manifest: Manifest,
    workspace_id: str,
    settings: Settings,
    only_repo: str | None = None,
    embed_provider: str | None = None,
    incremental: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline. Returns a stats dict for the CLI.

    Set ``incremental=True`` to use per-file BLAKE3 diff against Neo4j: only
    changed files are reparsed and written; deleted files are surgically
    dropped. Cross-service flows are always recomputed workspace-wide.
    """
    workspace = manifest.workspace(workspace_id)
    m_sha = manifest_sha(manifest)

    # 1-3. Checkout + parse each selected repo.
    bundle = WorkspaceParseBundle(
        workspace_id=workspace.id,
        workspace_name=workspace.name,
        manifest_sha=m_sha,
    )
    repo_metadata: dict[str, dict[str, Any]] = {}
    diffs_by_repo: dict[str, FileDiff] = {}
    parse_started = time.perf_counter()

    existing_hashes_by_repo: dict[str, dict[str, str]] = {}
    if incremental:
        with Neo4jClient(settings.neo4j) as client:
            for repo in workspace.repos:
                if only_repo and repo.name != only_repo:
                    continue
                existing_hashes_by_repo[repo.name] = fetch_existing_hashes(
                    client=client,
                    workspace_id=workspace.id,
                    repo_name=repo.name,
                )

    with stage_timer("parse", workspace.id):
        for repo in workspace.repos:
            if only_repo and repo.name != only_repo:
                continue
            pr, meta, diff = _parse_repo(
                workspace=workspace,
                repo=repo,
                settings=settings,
                existing_hashes=existing_hashes_by_repo.get(repo.name) if incremental else None,
            )
            bundle.repos.append(pr)
            repo_metadata[repo.name] = meta
            if diff is not None:
                diffs_by_repo[repo.name] = diff
            _update_parse_metrics(workspace.id, pr)

    parse_elapsed = time.perf_counter() - parse_started
    _log.info(
        "workspace parsed",
        workspace=workspace.id,
        repos=len(bundle.repos),
        files=sum(len(p.files) for p in bundle.repos),
        elapsed_s=round(parse_elapsed, 3),
    )

    if not bundle.repos:
        _log.warning("nothing to index (no repos matched)", workspace=workspace.id)
        return {"workspace": workspace.id, "repos": []}

    # 4. Resolver — workspace-wide call resolution.
    with stage_timer("resolve", workspace.id):
        resolver_stats = resolve_workspace(bundle)
    for strategy, n in resolver_stats.items():
        METRICS.indexer_resolved_calls.labels(workspace=workspace.id, strategy=strategy).inc(n)

    # 5-6. Cross-service flows.
    with stage_timer("http_flows", workspace.id):
        bundle.http_flows.extend(match_http_flows(bundle))
    METRICS.indexer_http_flows.labels(workspace=workspace.id).inc(len(bundle.http_flows))
    with stage_timer("kafka_flows", workspace.id):
        bundle.kafka_flows.extend(match_kafka_flows(bundle))
    METRICS.indexer_kafka_flows.labels(workspace=workspace.id).inc(len(bundle.kafka_flows))

    # 7-8. Chunking happens inline during parse (see `_walk_repo`); here we
    # just summarise. Embedding is its own stage, parallelised per-repo.
    chunker_stats = _summarise_chunks(bundle)
    _log.info(
        "workspace chunked",
        workspace=workspace.id,
        chunks=chunker_stats["chunks"],
    )
    if embed_provider not in (None, "none", "off"):
        provider = _resolve_embedding_provider(embed_provider, settings=settings)
        if provider is not None:
            with stage_timer("embed", workspace.id):
                _embed_workspace(
                    bundle=bundle,
                    provider=provider,
                    expected_dim=settings.embedding.dim,
                    workers=settings.embedding.workers,
                    batch_size=settings.embedding.batch_size,
                )

    # 9. Write.
    write_started = time.perf_counter()
    with stage_timer("write", workspace.id), Neo4jClient(settings.neo4j) as client:
        writer = GraphWriter(client)
        if incremental:
            deletions_by_repo = {
                name: sorted(diff.deleted_paths) for name, diff in diffs_by_repo.items()
            }
            write_stats = writer.write_workspace_bundle_delta(
                bundle=bundle,
                repo_metadata=repo_metadata,
                deletions_by_repo=deletions_by_repo,
            )
        else:
            write_stats = writer.write_workspace_bundle(bundle=bundle, repo_metadata=repo_metadata)
    write_elapsed = time.perf_counter() - write_started

    return {
        "workspace": workspace.id,
        "manifest_sha": m_sha,
        "mode": "incremental" if incremental else "full",
        "repos": [
            {
                "repo": pr.repo_name,
                "commit": pr.commit_sha,
                **write_stats["repos"].get(pr.repo_name, {}),
                **(
                    {
                        "diff": {
                            "unchanged": len(diffs_by_repo[pr.repo_name].unchanged_paths),
                            "changed": len(diffs_by_repo[pr.repo_name].changed_paths),
                            "deleted": len(diffs_by_repo[pr.repo_name].deleted_paths),
                        }
                    }
                    if pr.repo_name in diffs_by_repo
                    else {}
                ),
            }
            for pr in bundle.repos
        ],
        "resolver": resolver_stats,
        "chunker": chunker_stats,
        "http_flows": write_stats["http_flows"],
        "kafka_flows": write_stats["kafka_flows"],
        "elapsed_s": {
            "parse": round(parse_elapsed, 3),
            "write": round(write_elapsed, 3),
        },
    }


def _update_parse_metrics(workspace_id: str, pr: ParseResult) -> None:
    """Bump per-repo Prometheus counters after a successful parse."""
    files_by_lang: dict[str, int] = {}
    symbols_by_kind: dict[str, int] = {}
    for f in pr.files:
        files_by_lang[f.lang] = files_by_lang.get(f.lang, 0) + 1
        for s in f.symbols:
            kind = str(s.kind)
            symbols_by_kind[kind] = symbols_by_kind.get(kind, 0) + 1
    for lang, n in files_by_lang.items():
        METRICS.indexer_files_parsed.labels(
            workspace=workspace_id, repo=pr.repo_name, lang=lang
        ).inc(n)
    for kind, n in symbols_by_kind.items():
        METRICS.indexer_symbols_emitted.labels(
            workspace=workspace_id, repo=pr.repo_name, kind=kind
        ).inc(n)
    chunk_count = sum(len(f.chunks) for f in pr.files)
    if chunk_count:
        METRICS.indexer_chunks.labels(workspace=workspace_id, repo=pr.repo_name).inc(chunk_count)


# ----------------------------------------------------------------- per-repo


def _parse_repo(
    *,
    workspace: WorkspaceConfig,
    repo: RepoConfig,
    settings: Settings,
    existing_hashes: dict[str, str] | None = None,
) -> tuple[ParseResult, dict[str, Any], FileDiff | None]:
    """Materialise + parse one repo. When ``existing_hashes`` is supplied the
    walker computes a :class:`FileDiff` and only parses changed files."""
    started = time.perf_counter()
    _log.info("indexing repo", workspace=workspace.id, repo=repo.name)

    checkout = ensure_checkout(
        workspace_id=workspace.id,
        repo=repo,
        checkout_root=settings.checkout_root,
    )

    diff: FileDiff | None = None
    walk_kwargs: dict[str, Any] = {
        "workspace_id": workspace.id,
        "checkout": checkout,
        "repo": repo,
        "max_file_bytes": settings.max_file_bytes,
    }
    if existing_hashes is not None:
        on_disk = list(
            _iter_repo_source_pairs(
                checkout=checkout,
                repo=repo,
                max_file_bytes=settings.max_file_bytes,
            )
        )
        diff = compute_diff(
            workspace_id=workspace.id,
            repo_name=repo.name,
            existing_hashes=existing_hashes,
            on_disk_files=on_disk,
        )
        walk_kwargs["only_paths"] = diff.changed_paths
        file_records = list(_walk_repo(**walk_kwargs))
    else:
        file_records = list(_walk_repo(**walk_kwargs))

    pr = ParseResult(
        workspace_id=workspace.id,
        repo_name=repo.name,
        commit_sha=checkout.commit_sha,
        files=file_records,
    )
    elapsed = time.perf_counter() - started
    _log.info(
        "repo parsed",
        repo=repo.name,
        commit=checkout.commit_sha,
        files=len(file_records),
        elapsed_s=round(elapsed, 3),
        mode="incremental" if existing_hashes is not None else "full",
    )
    return pr, {"url": repo.url, "path": repo.path, "branch": repo.branch}, diff


def _walk_repo(
    *,
    workspace_id: str,
    checkout: CheckoutResult,
    repo: RepoConfig,
    max_file_bytes: int,
    only_paths: frozenset[str] | None = None,
) -> Iterable[FileRecord]:
    """Walk a checkout, parse + chunk each source file, drop the bytes.

    This used to populate a workspace-wide source cache and let a later
    ``_chunk_workspace`` stage re-read every file. That design held every
    repo's raw bytes in RAM through the resolver + flow-matcher stages —
    catastrophic on a 4 GiB box with even one 10 MiB minified bundle in the
    workspace. We now:

    * **Skip files larger than ``max_file_bytes``** before reading them.
      Any source over the cap is logged + counted but never opened.
    * **Chunk inline** right after parsing each file and attach the chunks
      to the returned :class:`FileRecord`.
    * **Drop the source bytes** as soon as parsing + chunking is done so
      memory pressure stays bounded to one file at a time.

    If ``only_paths`` is provided (incremental mode), only files whose
    repo-relative path is in the set are parsed; everything else is skipped.
    """
    enabled = set(repo.languages)
    adapters = {lang: get_adapter(lang) for lang in enabled}
    skipped_oversize = 0
    biggest_skipped = 0

    for source_path in _iter_source_files(checkout.root):
        rel = source_path.relative_to(checkout.root).as_posix()
        if only_paths is not None and rel not in only_paths:
            continue
        lang = _detect_language(source_path, adapters)
        if lang is None:
            continue

        # Cheap stat() *before* reading the bytes — keeps oversize files
        # from ever entering the process address space.
        try:
            size = source_path.stat().st_size
        except OSError as exc:
            _log.warning("cannot stat file", path=rel, error=str(exc))
            continue
        if max_file_bytes > 0 and size > max_file_bytes:
            skipped_oversize += 1
            biggest_skipped = max(biggest_skipped, size)
            _log.warning(
                "skipping oversize file (above CODE_SPIDER_MAX_FILE_BYTES)",
                repo=repo.name,
                path=rel,
                size_bytes=size,
                cap_bytes=max_file_bytes,
            )
            continue

        adapter = adapters[lang]
        try:
            source = source_path.read_bytes()
        except OSError as exc:
            _log.warning("cannot read file", path=rel, error=str(exc))
            continue

        try:
            record = adapter.parse_file(rel, source)
        except Exception as exc:
            # We never want one bad file to kill the run.
            _log.error("parse failure", path=rel, error=str(exc), lang=lang)
            continue

        hash_hex = blake3.blake3(source).hexdigest()
        record = replace(record, hash_blake3=hash_hex)

        # Chunk inline. Doing this here (instead of in a separate workspace
        # pass) lets us free ``source`` immediately and keeps the resident
        # set bounded to one file's worth of bytes at any moment.
        try:
            chunks = chunk_file(
                workspace_id=workspace_id,
                repo_name=repo.name,
                file_record=record,
                source=source,
            )
            record.chunks.extend(chunks)
        except Exception as exc:
            _log.error(
                "chunker failure (continuing without chunks for this file)",
                path=rel,
                error=str(exc),
            )

        # ``source`` goes out of scope at the next loop iteration; explicit
        # `del` documents intent and helps when this loop is profiled.
        del source
        yield record

    if skipped_oversize:
        _log.info(
            "walker skipped oversize files",
            repo=repo.name,
            skipped=skipped_oversize,
            biggest_bytes=biggest_skipped,
            cap_bytes=max_file_bytes,
        )


def _iter_repo_source_pairs(
    *, checkout: CheckoutResult, repo: RepoConfig, max_file_bytes: int
) -> Iterable[tuple[str, bytes]]:
    """Yield ``(repo_relative_path, source_bytes)`` for every supported file.

    Used by the incremental-diff path. Honors the same size cap as
    :func:`_walk_repo` so oversize files never participate in BLAKE3
    hashing.
    """
    enabled = set(repo.languages)
    adapters = {lang: get_adapter(lang) for lang in enabled}
    for source_path in _iter_source_files(checkout.root):
        rel = source_path.relative_to(checkout.root).as_posix()
        if _detect_language(source_path, adapters) is None:
            continue
        try:
            size = source_path.stat().st_size
        except OSError as exc:
            _log.warning("cannot stat file (diff)", path=rel, error=str(exc))
            continue
        if max_file_bytes > 0 and size > max_file_bytes:
            continue
        try:
            yield rel, source_path.read_bytes()
        except OSError as exc:
            _log.warning("cannot read file (diff)", path=rel, error=str(exc))


def _iter_source_files(root: Path) -> Iterable[Path]:
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in _IGNORED_DIRS:
                    continue
                stack.append(entry)
            elif entry.is_file():
                yield entry


def _detect_language(path: Path, adapters: dict[str, object]) -> str | None:
    suffix = path.suffix
    for lang, adapter in adapters.items():
        if suffix in getattr(adapter, "extensions", ()):
            return lang
    return None


# --------------------------------------------------------------- chunk/embed


def _summarise_chunks(bundle: WorkspaceParseBundle) -> dict[str, int]:
    """Sum chunk counts across every repo for the stats payload.

    Chunks are produced inline by :func:`_walk_repo` to keep source bytes
    out of memory; this helper just rolls up the totals.
    """
    total = 0
    for pr in bundle.repos:
        for f in pr.files:
            total += len(f.chunks)
    return {"chunks": total}


def _resolve_embedding_provider(
    name: str | None, *, settings: Settings | None = None
) -> EmbeddingProvider | None:
    """Resolve a provider, gracefully degrading if the model is unavailable.

    Resolution rules:

    * ``name in (None, "auto")`` reads ``settings.embedding.provider`` (which
      itself comes from ``CODE_SPIDER_EMBED_PROVIDER``, default
      ``sentence-transformers``). The CLI flag is left at ``auto`` so the
      env-driven default takes effect; an explicit flag value always wins.
    * Any other ``name`` is resolved directly from the registry. For
      ``"litellm"`` the settings bundle is required so the LiteLLM adapter
      can read the configured model / API base / etc.
    """
    embed_settings = settings.embedding if settings is not None else None

    if name in (None, "auto"):
        target = embed_settings.provider if embed_settings else "sentence-transformers"
    else:
        target = name

    try:
        return get_embedding_provider(target, settings=embed_settings)
    except ImportError as exc:
        _log.warning(
            "embedding provider unavailable; chunks will be persisted without "
            "embeddings (vector search disabled).",
            provider=target,
            error=str(exc),
        )
        return None
    except (KeyError, ValueError) as exc:
        _log.error("could not resolve embedding provider", name=target, error=str(exc))
        return None


def _embed_workspace(
    *,
    bundle: WorkspaceParseBundle,
    provider: EmbeddingProvider,
    expected_dim: int | None = None,
    workers: int = 1,
    batch_size: int = 64,
) -> None:
    """Compute embeddings for every chunk in the bundle.

    Strategy:

    * **Per-repo isolation** — each repo is processed independently. If a
      repo's embed raises (provider outage, persistent oversize chunk), the
      failure is logged and the loop continues with the next repo so the
      workspace still gets written.
    * **Parallel sub-batches per repo** — within a repo, chunk texts are
      split into ``batch_size`` slices and submitted to a
      :class:`ThreadPoolExecutor` with ``workers`` threads. Embedding work
      is I/O-bound (network calls to the provider) so threading — not
      multiprocessing — is the right fit and avoids GIL contention.
    * **Live progress** — when stderr is a TTY, a :mod:`rich.progress` bar
      tracks chunks completed / total / rate / ETA per workspace. When not
      a TTY (CI, ``code-spider serve`` under an agent), the same data is
      logged structurally every ~5 % so users still see the work breathing.

    The full bundle (including any repos whose embeddings failed and were
    written without vectors) is persisted in a single workspace-level write
    at the end of :func:`index_workspace`. Chunks that failed to embed are
    still indexed structurally; only vector search loses coverage for them.
    """
    if not bundle.repos:
        return

    # Fail fast on dim mismatch so an operator who flipped CODE_SPIDER_EMBED_*
    # without re-migrating gets a clear actionable error instead of a
    # confusing Neo4j vector-index rejection downstream.
    if expected_dim is not None and provider.dim != expected_dim:
        raise RuntimeError(
            f"embedding provider '{provider.name}' reports dim={provider.dim} "
            f"but CODE_SPIDER_EMBED_DIM={expected_dim}. Re-run "
            "`code-spider migrate` after updating CODE_SPIDER_EMBED_DIM to "
            "match your chosen model (drop the existing chunk_embedding index "
            "first)."
        )

    total_chunks = sum(len(f.chunks) for pr in bundle.repos for f in pr.files)
    if total_chunks == 0:
        return

    workers = max(1, workers)
    batch_size = max(1, batch_size)
    _log.info(
        "embedding workspace",
        workspace=bundle.workspace_id,
        chunks=total_chunks,
        repos=len(bundle.repos),
        workers=workers,
        batch_size=batch_size,
        provider=provider.name,
        dim=provider.dim,
    )

    total_embedded = 0
    total_failed_repos = 0
    with _embed_progress(total_chunks=total_chunks, workspace_id=bundle.workspace_id) as progress:
        for r_idx in range(len(bundle.repos)):
            ok = _embed_one_repo(
                bundle=bundle,
                r_idx=r_idx,
                provider=provider,
                workers=workers,
                batch_size=batch_size,
                progress=progress,
            )
            if ok is None:
                total_failed_repos += 1
            else:
                total_embedded += ok

    _log.info(
        "workspace embedded",
        workspace=bundle.workspace_id,
        chunks=total_embedded,
        failed_repos=total_failed_repos,
        provider=provider.name,
        dim=provider.dim,
    )


def _embed_one_repo(
    *,
    bundle: WorkspaceParseBundle,
    r_idx: int,
    provider: EmbeddingProvider,
    workers: int,
    batch_size: int,
    progress: _ProgressReporter,
) -> int | None:
    """Embed every chunk of one repo in parallel sub-batches.

    Returns the number of chunks successfully embedded, or ``None`` if the
    repo failed entirely (caller bumps a per-workspace failure counter).
    """
    pr = bundle.repos[r_idx]
    pairs: list[tuple[int, int]] = []  # (file_idx, chunk_idx)
    texts: list[str] = []
    for f_idx, f in enumerate(pr.files):
        for c_idx, c in enumerate(f.chunks):
            pairs.append((f_idx, c_idx))
            texts.append(c.text)
    if not texts:
        return 0

    progress.start_repo(pr.repo_name, total=len(texts))
    started = time.perf_counter()

    # Split into batch-sized sub-batches; submit to a thread pool. We track
    # original (start, end) so results land in the correct slots even when
    # futures complete out-of-order.
    slices: list[tuple[int, int]] = []
    for start in range(0, len(texts), batch_size):
        slices.append((start, min(start + batch_size, len(texts))))

    vectors, n_failures = _dispatch_slices(
        provider=provider,
        texts=texts,
        slices=slices,
        workers=workers,
        on_advance=progress.advance_repo,
        log_ctx={"workspace": bundle.workspace_id, "repo": pr.repo_name},
    )

    elapsed = time.perf_counter() - started
    progress.finish_repo()

    if n_failures == len(slices):
        _log.error(
            "every sub-batch failed for repo; embeddings unavailable",
            workspace=bundle.workspace_id,
            repo=pr.repo_name,
            sub_batches=len(slices),
        )
        return None

    # Apply the successful vectors. Slots that stayed ``None`` (failed
    # sub-batches) keep their existing empty embedding tuple — those chunks
    # are still indexed structurally; only vector search loses them.
    applied = 0
    for (f_idx, c_idx), vec in zip(pairs, vectors, strict=True):
        if vec is None:
            continue
        chunks = bundle.repos[r_idx].files[f_idx].chunks
        chunks[c_idx] = replace(chunks[c_idx], embedding=tuple(vec))
        applied += 1

    rate = applied / elapsed if elapsed > 0 else 0.0
    _log.info(
        "repo embedded",
        workspace=bundle.workspace_id,
        repo=pr.repo_name,
        chunks=applied,
        failed_sub_batches=n_failures,
        elapsed_s=round(elapsed, 2),
        rate_per_s=round(rate, 1),
        provider=provider.name,
    )
    return applied


def _dispatch_slices(
    *,
    provider: EmbeddingProvider,
    texts: list[str],
    slices: list[tuple[int, int]],
    workers: int,
    on_advance: Any,
    log_ctx: dict[str, Any],
) -> tuple[list[list[float] | None], int]:
    """Embed each slice (single-thread fast path or thread-pool) and
    aggregate results into one ``vectors`` array.

    Pulled out of :func:`_embed_one_repo` so the branch-count of the main
    function stays under ruff's threshold; also lets us unit-test the
    dispatcher in isolation if we ever need to.
    """
    vectors: list[list[float] | None] = [None] * len(texts)
    failures = 0

    def _land(
        start: int,
        end: int,
        vecs: list[list[float]] | None,
        exc: Exception | None,
    ) -> None:
        nonlocal failures
        ok = exc is None and vecs is not None and len(vecs) == end - start
        if ok:
            assert vecs is not None
            for i, vec in enumerate(vecs):
                vectors[start + i] = vec
        else:
            failures += 1
            _log.error(
                "sub-batch embed failed",
                **log_ctx,
                range=f"{start}..{end}",
                expected=end - start,
                got=0 if vecs is None else len(vecs),
                error=str(exc) if exc is not None else "wrong-shape response",
            )
        on_advance(end - start)

    # Single-thread fast path: keep tests deterministic, skip executor cost.
    if workers == 1 or len(slices) == 1:
        for start, end in slices:
            try:
                _land(start, end, provider.embed_batch(texts[start:end]), None)
            except Exception as exc:
                _land(start, end, None, exc)
        return vectors, failures

    # Parallel path: threaded I/O. Bounded by `workers`.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed") as pool:
        future_to_slice = {pool.submit(provider.embed_batch, texts[s:e]): (s, e) for s, e in slices}
        for future in as_completed(future_to_slice):
            start, end = future_to_slice[future]
            try:
                _land(start, end, future.result(), None)
            except Exception as exc:
                _land(start, end, None, exc)
    return vectors, failures
