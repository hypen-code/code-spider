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
    _SOURCE_CACHE.clear()

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

    # 7-8. Chunking + (optional) embeddings.
    with stage_timer("chunk", workspace.id):
        chunker_stats = _chunk_workspace(bundle=bundle)
    if embed_provider not in (None, "none", "off"):
        provider = _resolve_embedding_provider(embed_provider, settings=settings)
        if provider is not None:
            with stage_timer("embed", workspace.id):
                _embed_workspace(
                    bundle=bundle,
                    provider=provider,
                    expected_dim=settings.embedding.dim,
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
            write_stats = writer.write_workspace_bundle(
                bundle=bundle, repo_metadata=repo_metadata
            )
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
    if existing_hashes is not None:
        on_disk = list(_iter_repo_source_pairs(checkout=checkout, repo=repo))
        diff = compute_diff(
            workspace_id=workspace.id,
            repo_name=repo.name,
            existing_hashes=existing_hashes,
            on_disk_files=on_disk,
        )
        changed_set = diff.changed_paths
        file_records = list(
            _walk_repo(
                checkout=checkout, repo=repo, only_paths=changed_set
            )
        )
    else:
        file_records = list(_walk_repo(checkout=checkout, repo=repo))

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
    checkout: CheckoutResult,
    repo: RepoConfig,
    only_paths: frozenset[str] | None = None,
) -> Iterable[FileRecord]:
    """Walk a checkout, parse each source file, populate the source cache.

    If ``only_paths`` is provided, only files whose repo-relative path is in
    the set are parsed; everything else is skipped. The cache lets
    :func:`_chunk_workspace` re-read bytes without a second disk hit.
    """
    enabled = set(repo.languages)
    adapters = {lang: get_adapter(lang) for lang in enabled}

    for source_path in _iter_source_files(checkout.root):
        rel = source_path.relative_to(checkout.root).as_posix()
        if only_paths is not None and rel not in only_paths:
            continue
        lang = _detect_language(source_path, adapters)
        if lang is None:
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
        _SOURCE_CACHE[(checkout.repo_name, rel)] = source
        yield replace(record, hash_blake3=hash_hex)


def _iter_repo_source_pairs(
    *, checkout: CheckoutResult, repo: RepoConfig
) -> Iterable[tuple[str, bytes]]:
    """Yield ``(repo_relative_path, source_bytes)`` for every supported file."""
    enabled = set(repo.languages)
    adapters = {lang: get_adapter(lang) for lang in enabled}
    for source_path in _iter_source_files(checkout.root):
        rel = source_path.relative_to(checkout.root).as_posix()
        if _detect_language(source_path, adapters) is None:
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


def _chunk_workspace(*, bundle: WorkspaceParseBundle) -> dict[str, int]:
    """Populate ``FileRecord.chunks`` for every file in the bundle."""
    total = 0
    for pr in bundle.repos:
        for f in pr.files:
            chunks = chunk_file(
                workspace_id=bundle.workspace_id,
                repo_name=pr.repo_name,
                file_record=f,
                source=_read_source_for(pr, f),
            )
            f.chunks.extend(chunks)
            total += len(chunks)
    _log.info("workspace chunked", chunks=total)
    return {"chunks": total}


def _read_source_for(pr: ParseResult, f: FileRecord) -> bytes:
    """Return the raw bytes for a parsed file (chunker-side accessor)."""
    return _SOURCE_CACHE[(pr.repo_name, f.repo_relative_path)]


# Single-process bytes cache keyed by ``(repo_name, repo_relative_path)``.
# Populated during the parse walk so the chunker doesn't re-read from disk.
# Cleared at the start of every :func:`index_workspace` run.
_SOURCE_CACHE: dict[tuple[str, str], bytes] = {}


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
) -> None:
    """Compute embeddings for every chunk in the bundle."""
    pairs: list[tuple[int, int, int]] = []  # (repo_idx, file_idx, chunk_idx)
    texts: list[str] = []
    for r_idx, pr in enumerate(bundle.repos):
        for f_idx, f in enumerate(pr.files):
            for c_idx, c in enumerate(f.chunks):
                pairs.append((r_idx, f_idx, c_idx))
                texts.append(c.text)
    if not texts:
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

    vectors = provider.embed_batch(texts)
    if len(vectors) != len(pairs):
        _log.error(
            "embedding provider returned wrong shape",
            expected=len(pairs),
            got=len(vectors),
        )
        return

    for (r_idx, f_idx, c_idx), vec in zip(pairs, vectors, strict=True):
        chunks = bundle.repos[r_idx].files[f_idx].chunks
        chunks[c_idx] = replace(chunks[c_idx], embedding=tuple(vec))
    _log.info(
        "workspace embedded",
        chunks=len(pairs),
        provider=provider.name,
        dim=provider.dim,
    )


