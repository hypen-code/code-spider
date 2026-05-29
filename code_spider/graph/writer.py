"""Idempotent Neo4j writer for a full workspace bundle (Phase 1).

Strategy — **full workspace refresh** (Phase 1; Phase 2 swaps to incremental):

    1. For each repo in the bundle, delete every per-repo node (Commit, File,
       Module, Symbol, Route, HttpClientCall, KafkaProducer, KafkaConsumer,
       Chunk). KafkaTopic and Workspace/Repository persist across runs.
    2. UNWIND + MERGE every node in batches.
    3. Wire up structural edges (CONTAINS, HAS_COMMIT, DEFINES, ENCLOSES).
    4. Wire up feature edges (HANDLES, INVOKES, PRODUCES, CONSUMES,
       WRITES_TO, READS_FROM, IMPORTS).
    5. Wire up cross-service edges (HTTP_FLOW, KAFKA_FLOW) and resolved
       :CALLS edges from the resolver.

All operations use ``MERGE`` keyed on unique constraints (see
:mod:`code_spider.graph.schema`) so the writer is safe to retry mid-failure.

Public entry point: :meth:`GraphWriter.write_workspace_bundle`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from typing import Any

from neo4j import ManagedTransaction

from code_spider.graph.client import Neo4jClient, retry_on_transient
from code_spider.graph.count_cache import invalidate_workspace
from code_spider.logging_setup import get_logger
from code_spider.symbols.model import (
    Chunk,
    FileRecord,
    HttpClientCall,
    HttpFlowEdge,
    KafkaConsumer,
    KafkaFlowEdge,
    KafkaProducer,
    ParseResult,
    ResolvedCall,
    Route,
    Symbol,
    WorkspaceParseBundle,
)

_log = get_logger(__name__)


_SYMBOL_KIND_TO_LABEL: dict[str, str] = {
    "function": "Function",
    "method": "Method",
    "class": "Class",
    "interface": "Interface",
    "type_alias": "TypeAlias",
    "variable": "Variable",
}


# ----------------------------------------------------------- Cypher constants
# Each is parameterised; UNWIND parameters use ``$batch``.

_CLEAR_REPO = """
MATCH (r:Repository {workspace_id: $workspace_id, name: $repo_name})
OPTIONAL MATCH (r)-[:HAS_COMMIT]->(c:Commit)
OPTIONAL MATCH (c)-[:CONTAINS]->(f:File)
OPTIONAL MATCH (f)-[:DEFINES]->(s:Symbol)
OPTIONAL MATCH (f)-[:HAS_CHUNK]->(ck:Chunk)
DETACH DELETE c, f, s, ck
"""

_CLEAR_REPO_MODULES = """
MATCH (m:Module {workspace_id: $workspace_id, repo: $repo_name}) DETACH DELETE m
"""

_CLEAR_REPO_ROUTES_AND_CLIENTS = """
MATCH (n {workspace_id: $workspace_id, repo: $repo_name})
WHERE n:Route OR n:HttpClientCall OR n:KafkaProducer OR n:KafkaConsumer
DETACH DELETE n
"""

_UPSERT_WORKSPACE = """
MERGE (w:Workspace {id: $workspace_id})
SET w.name = coalesce($workspace_name, w.name),
    w.manifest_sha = $manifest_sha,
    w.updated_at = datetime()
"""

_UPSERT_REPO = """
MERGE (r:Repository {workspace_id: $workspace_id, name: $repo_name})
SET r.url = $url,
    r.path = $path,
    r.branch = $branch,
    r.last_indexed_sha = $commit_sha,
    r.updated_at = datetime()
WITH r
MATCH (w:Workspace {id: $workspace_id})
MERGE (w)-[:CONTAINS]->(r)
"""

_UPSERT_COMMIT = """
MATCH (r:Repository {workspace_id: $workspace_id, name: $repo_name})
MERGE (c:Commit {workspace_id: $workspace_id, repo: $repo_name, sha: $commit_sha})
SET c.indexed_at = datetime()
MERGE (r)-[:HAS_COMMIT]->(c)
"""

_UPSERT_FILES = """
MATCH (c:Commit {workspace_id: $workspace_id, repo: $repo_name, sha: $commit_sha})
UNWIND $batch AS f
MERGE (file:File {workspace_id: $workspace_id, repo: $repo_name, path: f.path})
SET file.lang = f.lang,
    file.hash_blake3 = f.hash_blake3,
    file.size = f.size,
    file.loc = f.loc,
    file.commit_sha = $commit_sha
MERGE (c)-[:CONTAINS]->(file)
"""

_UPSERT_MODULES = """
UNWIND $batch AS m
MERGE (mod:Module {workspace_id: $workspace_id, repo: $repo_name, fqn: m.fqn})
SET mod.kind = m.kind, mod.file_path = m.file_path
WITH mod, m
MATCH (file:File {workspace_id: $workspace_id, repo: $repo_name, path: m.file_path})
MERGE (file)-[:DEFINES_MODULE]->(mod)
"""

_UPSERT_SYMBOLS = """
UNWIND $batch AS s
MERGE (sym:Symbol {workspace_id: $workspace_id, repo: $repo_name, fqn: s.fqn})
SET sym.name = s.name,
    sym.kind = s.kind,
    sym.lang = s.lang,
    sym.file_path = s.file_path,
    sym.start_line = s.start_line,
    sym.start_col = s.start_col,
    sym.end_line = s.end_line,
    sym.end_col = s.end_col,
    sym.signature = s.signature,
    sym.docstring = s.docstring,
    sym.visibility = s.visibility,
    sym.parent_fqn = s.parent_fqn,
    sym.commit_sha = $commit_sha
FOREACH (_ IN CASE WHEN s.kind = 'function'    THEN [1] ELSE [] END | SET sym:Function)
FOREACH (_ IN CASE WHEN s.kind = 'method'      THEN [1] ELSE [] END | SET sym:Method)
FOREACH (_ IN CASE WHEN s.kind = 'class'       THEN [1] ELSE [] END | SET sym:Class)
FOREACH (_ IN CASE WHEN s.kind = 'interface'   THEN [1] ELSE [] END | SET sym:Interface)
FOREACH (_ IN CASE WHEN s.kind = 'type_alias'  THEN [1] ELSE [] END | SET sym:TypeAlias)
FOREACH (_ IN CASE WHEN s.kind = 'variable'    THEN [1] ELSE [] END | SET sym:Variable)
WITH sym, s
MATCH (file:File {workspace_id: $workspace_id, repo: $repo_name, path: s.file_path})
MERGE (file)-[:DEFINES]->(sym)
WITH sym, s
OPTIONAL MATCH (parent:Symbol {workspace_id: $workspace_id, repo: $repo_name, fqn: s.parent_fqn})
FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END |
  MERGE (parent)-[:ENCLOSES]->(sym)
)
"""

_UPSERT_IMPORTS = """
UNWIND $batch AS i
MATCH (file:File {workspace_id: $workspace_id, repo: $repo_name, path: i.file_path})
WITH file, i
OPTIONAL MATCH (target:Symbol {workspace_id: $workspace_id, fqn: i.resolved_fqn})
WITH file, i, target
WHERE target IS NOT NULL
MERGE (file)-[r:IMPORTS]->(target)
SET r.raw = i.raw,
    r.resolved_fqn = i.resolved_fqn,
    r.local_name = i.local_name
"""

_UPSERT_ROUTES = """
UNWIND $batch AS r
MERGE (route:Route {route_id: r.route_id})
SET route.workspace_id = $workspace_id,
    route.repo         = $repo_name,
    route.method       = r.method,
    route.path         = r.path,
    route.framework    = r.framework,
    route.file_path    = r.file_path,
    route.start_line   = r.start_line,
    route.end_line     = r.end_line
WITH route, r
OPTIONAL MATCH (sym:Symbol {workspace_id: $workspace_id, repo: $repo_name, fqn: r.handler_fqn})
FOREACH (_ IN CASE WHEN sym IS NULL THEN [] ELSE [1] END |
  MERGE (sym)-[:HANDLES]->(route)
)
"""

_UPSERT_HTTP_CLIENTS = """
UNWIND $batch AS h
MERGE (call:HttpClientCall {call_id: h.call_id})
SET call.workspace_id    = $workspace_id,
    call.repo            = $repo_name,
    call.method          = h.method,
    call.path_template   = h.path_template,
    call.base_url_hint   = h.base_url_hint,
    call.file_path       = h.file_path,
    call.start_line      = h.start_line,
    call.end_line        = h.end_line
WITH call, h
OPTIONAL MATCH (sym:Symbol {workspace_id: $workspace_id, repo: $repo_name, fqn: h.caller_fqn})
FOREACH (_ IN CASE WHEN sym IS NULL THEN [] ELSE [1] END |
  MERGE (sym)-[:INVOKES]->(call)
)
"""

_UPSERT_KAFKA_PRODUCERS = """
UNWIND $batch AS p
MERGE (topic:KafkaTopic {workspace_id: $workspace_id, name: p.topic_name})
MERGE (prod:KafkaProducer {producer_id: p.producer_id})
SET prod.workspace_id  = $workspace_id,
    prod.repo          = $repo_name,
    prod.topic_name    = p.topic_name,
    prod.client_lib    = p.client_lib,
    prod.file_path     = p.file_path,
    prod.start_line    = p.start_line,
    prod.end_line      = p.end_line
MERGE (prod)-[:WRITES_TO]->(topic)
WITH prod, p
OPTIONAL MATCH (sym:Symbol {workspace_id: $workspace_id, repo: $repo_name, fqn: p.caller_fqn})
FOREACH (_ IN CASE WHEN sym IS NULL THEN [] ELSE [1] END |
  MERGE (sym)-[:PRODUCES]->(prod)
)
"""

_UPSERT_KAFKA_CONSUMERS = """
UNWIND $batch AS c
MERGE (topic:KafkaTopic {workspace_id: $workspace_id, name: c.topic_name})
MERGE (cons:KafkaConsumer {consumer_id: c.consumer_id})
SET cons.workspace_id  = $workspace_id,
    cons.repo          = $repo_name,
    cons.topic_name    = c.topic_name,
    cons.client_lib    = c.client_lib,
    cons.group_id      = c.group_id,
    cons.file_path     = c.file_path,
    cons.start_line    = c.start_line,
    cons.end_line      = c.end_line
MERGE (cons)-[:READS_FROM]->(topic)
WITH cons, c
OPTIONAL MATCH (sym:Symbol {workspace_id: $workspace_id, repo: $repo_name, fqn: c.caller_fqn})
FOREACH (_ IN CASE WHEN sym IS NULL THEN [] ELSE [1] END |
  MERGE (sym)-[:CONSUMES]->(cons)
)
"""

_UPSERT_CHUNKS = """
UNWIND $batch AS ch
MERGE (chunk:Chunk {chunk_id: ch.chunk_id})
SET chunk.workspace_id = $workspace_id,
    chunk.repo         = $repo_name,
    chunk.file_path    = ch.file_path,
    chunk.start_line   = ch.start_line,
    chunk.end_line     = ch.end_line,
    chunk.text         = ch.text,
    chunk.embedding    = ch.embedding
WITH chunk, ch
MATCH (file:File {workspace_id: $workspace_id, repo: $repo_name, path: ch.file_path})
MERGE (file)-[:HAS_CHUNK]->(chunk)
"""

_UPSERT_CALLS = """
UNWIND $batch AS c
MATCH (src:Symbol {workspace_id: $workspace_id, repo: $caller_repo, fqn: c.caller_fqn})
MATCH (dst:Symbol {workspace_id: $workspace_id, repo: c.callee_repo, fqn: c.callee_fqn})
MERGE (src)-[r:CALLS {call_site_line: c.start_line, file_path: c.file_path}]->(dst)
SET r.confidence = c.confidence,
    r.strategy   = c.strategy
"""

_CLEAR_HTTP_FLOWS = """
MATCH (h:HttpClientCall {workspace_id: $workspace_id})-[r:HTTP_FLOW]->()
DELETE r
"""

_UPSERT_HTTP_FLOWS = """
UNWIND $batch AS e
MATCH (call:HttpClientCall {workspace_id: $workspace_id, repo: e.client_repo,
                            file_path: e.client_file_path, start_line: e.client_start_line})
MATCH (route:Route {workspace_id: $workspace_id, repo: e.route_repo,
                    method: e.method, path: e.path_template})
MERGE (call)-[r:HTTP_FLOW]->(route)
SET r.match_score = e.match_score
"""

_CLEAR_KAFKA_FLOWS = """
MATCH (p:KafkaProducer {workspace_id: $workspace_id})-[r:KAFKA_FLOW]->()
DELETE r
"""

# Surgical per-file delete (Phase 2 incremental). Drops every node sourced from
# the listed files: their Symbols, Routes, HttpClientCalls, Kafka producers/
# consumers, Chunks, and the File node itself. Modules are repo-scoped — the
# incremental writer rebuilds the module set from the changed files only.
_DELETE_FILES = """
UNWIND $paths AS p
MATCH (f:File {workspace_id: $workspace_id, repo: $repo_name, path: p})
OPTIONAL MATCH (f)-[:DEFINES]->(sym:Symbol)
OPTIONAL MATCH (f)-[:HAS_CHUNK]->(ck:Chunk)
DETACH DELETE sym, ck, f
"""

_DELETE_FEATURES_FOR_FILES = """
UNWIND $paths AS p
MATCH (n {workspace_id: $workspace_id, repo: $repo_name, file_path: p})
WHERE n:Route OR n:HttpClientCall OR n:KafkaProducer OR n:KafkaConsumer
DETACH DELETE n
"""

_DELETE_REPO_MODULES_FOR_FILES = """
UNWIND $paths AS p
MATCH (m:Module {workspace_id: $workspace_id, repo: $repo_name, file_path: p})
DETACH DELETE m
"""

_UPSERT_KAFKA_FLOWS = """
UNWIND $batch AS e
MATCH (prod:KafkaProducer {workspace_id: $workspace_id, repo: e.producer_repo,
                           topic_name: e.topic_name})
MATCH (cons:KafkaConsumer {workspace_id: $workspace_id, repo: e.consumer_repo,
                           topic_name: e.topic_name})
MERGE (prod)-[r:KAFKA_FLOW {topic_name: e.topic_name}]->(cons)
"""


# ------------------------------------------------------------------- writer


class GraphWriter:
    """High-level Neo4j upserter for one workspace indexing run."""

    def __init__(self, client: Neo4jClient, *, batch_size: int = 500) -> None:
        self._client = client
        self._batch_size = batch_size

    # ----------------------------------------------------------- public API

    @retry_on_transient
    def delete_files(
        self, *, workspace_id: str, repo_name: str, paths: list[str]
    ) -> None:
        """Surgically drop every node sourced from ``paths``. Idempotent."""
        if not paths:
            return
        params = {
            "workspace_id": workspace_id,
            "repo_name": repo_name,
            "paths": paths,
        }
        with self._client.session() as session:
            session.execute_write(_run_unit, query=_DELETE_FEATURES_FOR_FILES, params=params)
            session.execute_write(_run_unit, query=_DELETE_REPO_MODULES_FOR_FILES, params=params)
            session.execute_write(_run_unit, query=_DELETE_FILES, params=params)
        _log.info(
            "files deleted from graph",
            workspace=workspace_id,
            repo=repo_name,
            count=len(paths),
        )
        invalidate_workspace(workspace_id, trigger="delete")

    @retry_on_transient
    def write_workspace_bundle_delta(
        self,
        *,
        bundle: WorkspaceParseBundle,
        repo_metadata: dict[str, dict[str, Any]],
        deletions_by_repo: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Incremental version of :meth:`write_workspace_bundle`.

        ``bundle.repos`` carries only *changed* files for each repo; the
        ``deletions_by_repo`` map lists paths to drop. Unchanged files keep
        their existing graph nodes untouched.

        Cross-service flows (HTTP_FLOW, KAFKA_FLOW) and resolver outputs are
        still applied workspace-wide because their inputs span every repo.
        """
        stats: dict[str, Any] = {"repos": {}, "deletions": {}}

        with self._client.session() as session:
            for pr in bundle.repos:
                deletions = deletions_by_repo.get(pr.repo_name, [])
                if deletions:
                    self.delete_files(
                        workspace_id=bundle.workspace_id,
                        repo_name=pr.repo_name,
                        paths=deletions,
                    )
                    stats["deletions"][pr.repo_name] = len(deletions)

                self._write_repo_delta(
                    session=session,
                    bundle=bundle,
                    pr=pr,
                    repo_meta=repo_metadata[pr.repo_name],
                    out_stats=stats,
                )

            session.execute_write(
                _run_unit,
                query=_CLEAR_HTTP_FLOWS,
                params={"workspace_id": bundle.workspace_id},
            )
            session.execute_write(
                _run_unit,
                query=_CLEAR_KAFKA_FLOWS,
                params={"workspace_id": bundle.workspace_id},
            )

            for pr in bundle.repos:
                if not pr.resolved_calls:
                    continue
                self._batch_write(
                    session=session,
                    query=_UPSERT_CALLS,
                    items=(_resolved_call_payload(rc) for rc in pr.resolved_calls),
                    base_params={
                        "workspace_id": bundle.workspace_id,
                        "caller_repo": pr.repo_name,
                    },
                )

            for pr in bundle.repos:
                items = [
                    _import_payload(f.repo_relative_path, imp)
                    for f in pr.files
                    for imp in f.imports
                    if imp.resolved_fqn
                ]
                if not items:
                    continue
                self._batch_write(
                    session=session,
                    query=_UPSERT_IMPORTS,
                    items=iter(items),
                    base_params={
                        "workspace_id": bundle.workspace_id,
                        "repo_name": pr.repo_name,
                    },
                )

            if bundle.http_flows:
                self._batch_write(
                    session=session,
                    query=_UPSERT_HTTP_FLOWS,
                    items=(_http_flow_payload(e) for e in bundle.http_flows),
                    base_params={"workspace_id": bundle.workspace_id},
                )
            if bundle.kafka_flows:
                self._batch_write(
                    session=session,
                    query=_UPSERT_KAFKA_FLOWS,
                    items=(_kafka_flow_payload(e) for e in bundle.kafka_flows),
                    base_params={"workspace_id": bundle.workspace_id},
                )

        stats["http_flows"] = len(bundle.http_flows)
        stats["kafka_flows"] = len(bundle.kafka_flows)
        _log.info(
            "workspace delta write complete",
            workspace=bundle.workspace_id,
            **{k: v for k, v in stats.items() if k not in {"repos", "deletions"}},
        )
        invalidate_workspace(bundle.workspace_id, trigger="delta")
        return stats

    @retry_on_transient
    def write_workspace_bundle(
        self,
        *,
        bundle: WorkspaceParseBundle,
        repo_metadata: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist the whole workspace bundle. ``repo_metadata`` carries
        per-repo (url, path, branch). Returns a stats dict for the CLI."""
        stats: dict[str, Any] = {"repos": {}}

        with self._client.session() as session:
            # 1. Per-repo cleanup + structural writes.
            for pr in bundle.repos:
                self._write_repo(
                    session=session,
                    bundle=bundle,
                    pr=pr,
                    repo_meta=repo_metadata[pr.repo_name],
                    out_stats=stats,
                )

            # 2. Workspace-scoped edge cleanup (flows are re-derived).
            session.execute_write(
                _run_unit,
                query=_CLEAR_HTTP_FLOWS,
                params={"workspace_id": bundle.workspace_id},
            )
            session.execute_write(
                _run_unit,
                query=_CLEAR_KAFKA_FLOWS,
                params={"workspace_id": bundle.workspace_id},
            )

            # 3. Resolved CALLS edges (per-repo caller; callee may be cross-repo).
            for pr in bundle.repos:
                if not pr.resolved_calls:
                    continue
                self._batch_write(
                    session=session,
                    query=_UPSERT_CALLS,
                    items=(_resolved_call_payload(rc) for rc in pr.resolved_calls),
                    base_params={
                        "workspace_id": bundle.workspace_id,
                        "caller_repo": pr.repo_name,
                    },
                )

            # 4. Imports — resolved only (Neo4j MATCH on resolved_fqn).
            for pr in bundle.repos:
                items = [
                    _import_payload(f.repo_relative_path, imp)
                    for f in pr.files
                    for imp in f.imports
                    if imp.resolved_fqn
                ]
                if not items:
                    continue
                self._batch_write(
                    session=session,
                    query=_UPSERT_IMPORTS,
                    items=iter(items),
                    base_params={
                        "workspace_id": bundle.workspace_id,
                        "repo_name": pr.repo_name,
                    },
                )

            # 5. HTTP_FLOW edges (cross-service).
            if bundle.http_flows:
                self._batch_write(
                    session=session,
                    query=_UPSERT_HTTP_FLOWS,
                    items=(_http_flow_payload(e) for e in bundle.http_flows),
                    base_params={"workspace_id": bundle.workspace_id},
                )

            # 6. KAFKA_FLOW edges.
            if bundle.kafka_flows:
                self._batch_write(
                    session=session,
                    query=_UPSERT_KAFKA_FLOWS,
                    items=(_kafka_flow_payload(e) for e in bundle.kafka_flows),
                    base_params={"workspace_id": bundle.workspace_id},
                )

        stats["http_flows"] = len(bundle.http_flows)
        stats["kafka_flows"] = len(bundle.kafka_flows)
        _log.info(
            "workspace write complete",
            workspace=bundle.workspace_id,
            **{k: v for k, v in stats.items() if k != "repos"},
        )
        invalidate_workspace(bundle.workspace_id, trigger="full")
        return stats

    # --------------------------------------------------------- repo writes

    def _write_repo(
        self,
        *,
        session,
        bundle: WorkspaceParseBundle,
        pr: ParseResult,
        repo_meta: dict[str, Any],
        out_stats: dict[str, Any],
    ) -> None:
        """Full-refresh write: clear the repo's nodes then upsert everything."""
        common = _repo_common(bundle, pr, repo_meta)
        session.execute_write(_run_unit, query=_CLEAR_REPO, params=common)
        session.execute_write(_run_unit, query=_CLEAR_REPO_MODULES, params=common)
        session.execute_write(
            _run_unit, query=_CLEAR_REPO_ROUTES_AND_CLIENTS, params=common
        )
        self._write_repo_content(
            session=session, bundle=bundle, pr=pr, common=common, out_stats=out_stats
        )

    def _write_repo_delta(
        self,
        *,
        session,
        bundle: WorkspaceParseBundle,
        pr: ParseResult,
        repo_meta: dict[str, Any],
        out_stats: dict[str, Any],
    ) -> None:
        """Incremental write: assumes :meth:`delete_files` already dropped stale nodes."""
        common = _repo_common(bundle, pr, repo_meta)
        self._write_repo_content(
            session=session, bundle=bundle, pr=pr, common=common, out_stats=out_stats
        )

    def _write_repo_content(
        self,
        *,
        session,
        bundle: WorkspaceParseBundle,
        pr: ParseResult,
        common: dict[str, Any],
        out_stats: dict[str, Any],
    ) -> None:
        """Shared structural upsert path used by both full + delta writers."""
        repo_stats = {
            "files": 0,
            "modules": 0,
            "symbols": 0,
            "routes": 0,
            "http_clients": 0,
            "kafka_producers": 0,
            "kafka_consumers": 0,
            "chunks": 0,
        }

        session.execute_write(_run_unit, query=_UPSERT_WORKSPACE, params=common)
        session.execute_write(_run_unit, query=_UPSERT_REPO, params=common)
        session.execute_write(_run_unit, query=_UPSERT_COMMIT, params=common)

        files_payload = [_file_payload(f) for f in pr.files]
        modules_payload = [_module_payload(f) for f in pr.files if f.module]
        symbols_payload = [_symbol_payload(s) for f in pr.files for s in f.symbols]
        routes_payload = [
            _route_payload(bundle.workspace_id, pr.repo_name, r)
            for f in pr.files
            for r in f.routes
        ]
        clients_payload = [
            _http_client_payload(bundle.workspace_id, pr.repo_name, h)
            for f in pr.files
            for h in f.http_clients
        ]
        producers_payload = [
            _kafka_producer_payload(bundle.workspace_id, pr.repo_name, p)
            for f in pr.files
            for p in f.kafka_producers
        ]
        consumers_payload = [
            _kafka_consumer_payload(bundle.workspace_id, pr.repo_name, c)
            for f in pr.files
            for c in f.kafka_consumers
        ]
        chunks_payload = [_chunk_payload(c) for f in pr.files for c in f.chunks]

        if files_payload:
            self._batch_write(
                session=session, query=_UPSERT_FILES, items=iter(files_payload), base_params=common
            )
            repo_stats["files"] = len(files_payload)
        if modules_payload:
            self._batch_write(
                session=session,
                query=_UPSERT_MODULES,
                items=iter(modules_payload),
                base_params=common,
            )
            repo_stats["modules"] = len(modules_payload)
        if symbols_payload:
            self._batch_write(
                session=session,
                query=_UPSERT_SYMBOLS,
                items=iter(symbols_payload),
                base_params=common,
            )
            repo_stats["symbols"] = len(symbols_payload)
        if routes_payload:
            self._batch_write(
                session=session,
                query=_UPSERT_ROUTES,
                items=iter(routes_payload),
                base_params=common,
            )
            repo_stats["routes"] = len(routes_payload)
        if clients_payload:
            self._batch_write(
                session=session,
                query=_UPSERT_HTTP_CLIENTS,
                items=iter(clients_payload),
                base_params=common,
            )
            repo_stats["http_clients"] = len(clients_payload)
        if producers_payload:
            self._batch_write(
                session=session,
                query=_UPSERT_KAFKA_PRODUCERS,
                items=iter(producers_payload),
                base_params=common,
            )
            repo_stats["kafka_producers"] = len(producers_payload)
        if consumers_payload:
            self._batch_write(
                session=session,
                query=_UPSERT_KAFKA_CONSUMERS,
                items=iter(consumers_payload),
                base_params=common,
            )
            repo_stats["kafka_consumers"] = len(consumers_payload)
        if chunks_payload:
            self._batch_write(
                session=session,
                query=_UPSERT_CHUNKS,
                items=iter(chunks_payload),
                base_params=common,
            )
            repo_stats["chunks"] = len(chunks_payload)

        out_stats["repos"][pr.repo_name] = repo_stats

    # --------------------------------------------------------- batching

    def _batch_write(
        self,
        *,
        session,
        query: str,
        items: Iterator[dict[str, Any]],
        base_params: dict[str, Any],
    ) -> None:
        batch: list[dict[str, Any]] = []
        for item in items:
            batch.append(item)
            if len(batch) >= self._batch_size:
                session.execute_write(
                    _run_unit, query=query, params={**base_params, "batch": batch}
                )
                batch = []
        if batch:
            session.execute_write(
                _run_unit, query=query, params={**base_params, "batch": batch}
            )


# --------------------------------------------------------- payload mappers


def _repo_common(
    bundle: WorkspaceParseBundle, pr: ParseResult, repo_meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "workspace_id": bundle.workspace_id,
        "workspace_name": bundle.workspace_name,
        "manifest_sha": bundle.manifest_sha,
        "repo_name": pr.repo_name,
        "url": repo_meta.get("url"),
        "path": repo_meta.get("path"),
        "branch": repo_meta.get("branch", "main"),
        "commit_sha": pr.commit_sha,
    }


def _file_payload(f: FileRecord) -> dict[str, Any]:
    return {
        "path": f.repo_relative_path,
        "lang": f.lang,
        "hash_blake3": f.hash_blake3,
        "size": f.size_bytes,
        "loc": f.line_count,
    }


def _module_payload(f: FileRecord) -> dict[str, Any]:
    assert f.module is not None
    return {
        "fqn": f.module.fqn,
        "kind": f.module.kind,
        "file_path": f.module.file_path or f.repo_relative_path,
    }


def _symbol_payload(s: Symbol) -> dict[str, Any]:
    base = asdict(s)
    span = base.pop("span")
    base["kind"] = str(s.kind)
    base.update(
        start_line=span["start_line"],
        start_col=span["start_col"],
        end_line=span["end_line"],
        end_col=span["end_col"],
    )
    base["kind_label"] = _SYMBOL_KIND_TO_LABEL.get(str(s.kind), "Symbol")
    return base


def _route_payload(workspace_id: str, repo: str, r: Route) -> dict[str, Any]:
    return {
        "route_id": _stable_id(
            "route", workspace_id, repo, r.file_path, r.span.start_line, r.method, r.path
        ),
        "method": r.method,
        "path": r.path,
        "framework": r.framework,
        "handler_fqn": r.handler_fqn,
        "file_path": r.file_path,
        "start_line": r.span.start_line,
        "end_line": r.span.end_line,
    }


def _http_client_payload(
    workspace_id: str, repo: str, h: HttpClientCall
) -> dict[str, Any]:
    return {
        "call_id": _stable_id(
            "httpc",
            workspace_id,
            repo,
            h.file_path,
            h.span.start_line,
            h.method,
            h.path_template,
            h.caller_fqn,
        ),
        "method": h.method,
        "path_template": h.path_template,
        "base_url_hint": h.base_url_hint,
        "caller_fqn": h.caller_fqn,
        "file_path": h.file_path,
        "start_line": h.span.start_line,
        "end_line": h.span.end_line,
    }


def _kafka_producer_payload(
    workspace_id: str, repo: str, p: KafkaProducer
) -> dict[str, Any]:
    return {
        "producer_id": _stable_id(
            "kp",
            workspace_id,
            repo,
            p.file_path,
            p.span.start_line,
            p.topic_name,
            p.caller_fqn,
        ),
        "topic_name": p.topic_name,
        "client_lib": p.client_lib,
        "caller_fqn": p.caller_fqn,
        "file_path": p.file_path,
        "start_line": p.span.start_line,
        "end_line": p.span.end_line,
    }


def _kafka_consumer_payload(
    workspace_id: str, repo: str, c: KafkaConsumer
) -> dict[str, Any]:
    return {
        "consumer_id": _stable_id(
            "kc",
            workspace_id,
            repo,
            c.file_path,
            c.span.start_line,
            c.topic_name,
            c.caller_fqn,
        ),
        "topic_name": c.topic_name,
        "client_lib": c.client_lib,
        "group_id": c.group_id,
        "caller_fqn": c.caller_fqn,
        "file_path": c.file_path,
        "start_line": c.span.start_line,
        "end_line": c.span.end_line,
    }


def _chunk_payload(c: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "file_path": c.file_path,
        "start_line": c.span.start_line,
        "end_line": c.span.end_line,
        "text": c.text,
        "embedding": list(c.embedding) if c.embedding else None,
    }


def _resolved_call_payload(rc: ResolvedCall) -> dict[str, Any]:
    return {
        "caller_fqn": rc.caller_fqn,
        "callee_fqn": rc.callee_fqn,
        "callee_repo": rc.callee_repo,
        "confidence": rc.confidence,
        "strategy": rc.strategy,
        "file_path": rc.file_path,
        "start_line": rc.span.start_line,
    }


def _import_payload(file_path: str, imp) -> dict[str, Any]:
    return {
        "file_path": file_path,
        "raw": imp.raw,
        "local_name": imp.local_name,
        "resolved_fqn": imp.resolved_fqn,
    }


def _http_flow_payload(e: HttpFlowEdge) -> dict[str, Any]:
    return {
        "client_repo": e.client_repo,
        "client_file_path": e.client_file_path,
        "client_start_line": e.client_span.start_line,
        "route_repo": e.route_repo,
        "method": e.method,
        "path_template": e.path_template,
        "match_score": e.match_score,
    }


def _kafka_flow_payload(e: KafkaFlowEdge) -> dict[str, Any]:
    return {
        "producer_repo": e.producer_repo,
        "consumer_repo": e.consumer_repo,
        "topic_name": e.topic_name,
    }


# ----------------------------------------------------------- internals


def _stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic short ID derived from the joined parts (used as MERGE key)."""
    joined = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return f"{prefix}_" + hashlib.blake2b(joined, digest_size=12).hexdigest()


def _chunks(items: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    bucket: list[dict[str, Any]] = []
    for item in items:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def _run_unit(tx: ManagedTransaction, *, query: str, params: dict[str, Any]) -> None:
    # Pass through the explicit ``parameters=`` dict rather than ``**params`` so
    # a payload key named ``query`` / ``parameters`` cannot collide with the
    # driver signature ``tx.run(query, parameters=None, **kwparameters)``.
    tx.run(query, parameters=params).consume()
