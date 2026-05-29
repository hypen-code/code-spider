"""Cross-service HTTP_FLOW matcher.

For each :class:`HttpClientCall` discovered anywhere in the workspace, score
it against every :class:`Route` and emit an :class:`HttpFlowEdge` for every
pair above the threshold. Per-client deduplication keeps only the best match
plus any ties; the writer later persists them as ``:HTTP_FLOW`` edges with a
``match_score`` property.
"""

from __future__ import annotations

from code_spider.logging_setup import get_logger
from code_spider.routes._common import normalize_method, path_similarity
from code_spider.symbols.model import (
    HttpClientCall,
    HttpFlowEdge,
    Route,
    WorkspaceParseBundle,
)

_log = get_logger(__name__)

#: Minimum :func:`path_similarity` for a client→route pair to be emitted.
DEFAULT_THRESHOLD = 0.7


def match_http_flows(
    bundle: WorkspaceParseBundle, *, threshold: float = DEFAULT_THRESHOLD
) -> list[HttpFlowEdge]:
    """Produce the workspace's HTTP_FLOW edges. Caller appends to ``bundle.http_flows``."""
    routes: list[tuple[str, Route]] = [
        (pr.repo_name, r) for pr in bundle.repos for f in pr.files for r in f.routes
    ]
    clients: list[tuple[str, HttpClientCall]] = [
        (pr.repo_name, c)
        for pr in bundle.repos
        for f in pr.files
        for c in f.http_clients
    ]
    if not routes or not clients:
        _log.info(
            "no http_flow matching needed",
            routes=len(routes),
            clients=len(clients),
        )
        return []

    edges: list[HttpFlowEdge] = []
    for client_repo, client in clients:
        best_score = 0.0
        candidates: list[HttpFlowEdge] = []
        for route_repo, route in routes:
            if not _methods_match(client.method, route.method):
                continue
            score = path_similarity(client.path_template, route.path)
            if score < threshold:
                continue
            edge = HttpFlowEdge(
                client_caller_fqn=client.caller_fqn,
                client_repo=client_repo,
                client_file_path=client.file_path,
                client_span=client.span,
                route_handler_fqn=route.handler_fqn,
                route_repo=route_repo,
                method=normalize_method(route.method) if route.method != "*" else client.method,
                path_template=route.path,
                match_score=round(score, 4),
            )
            if score > best_score:
                best_score = score
                candidates = [edge]
            elif score == best_score:
                candidates.append(edge)
        edges.extend(candidates)

    _log.info(
        "http_flow edges materialised",
        routes=len(routes),
        clients=len(clients),
        edges=len(edges),
    )
    return edges


def _methods_match(client_method: str, route_method: str) -> bool:
    if client_method == "*" or route_method == "*":
        return True
    return normalize_method(client_method) == normalize_method(route_method)
