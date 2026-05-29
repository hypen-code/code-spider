"""Python route + HTTP client extractor tests."""

from __future__ import annotations

from textwrap import dedent

from code_spider.parser import get_adapter
from code_spider.routes._common import normalize_path, path_similarity


def _parse(source: str, path: str = "app/api.py"):
    return get_adapter("python").parse_file(path, dedent(source).encode("utf-8"))


def test_fastapi_routes_emit_correct_method_path_handler() -> None:
    fr = _parse(
        """
        from fastapi import FastAPI

        app = FastAPI()

        @app.get('/users/{user_id}')
        def get_user(user_id: int):
            return user_id

        @app.post('/users')
        async def create_user(payload: dict):
            return payload
        """
    )
    routes = {(r.method, r.path, r.handler_fqn) for r in fr.routes}
    assert ("GET", "/users/{}", "app.api.get_user") in routes
    assert ("POST", "/users", "app.api.create_user") in routes


def test_flask_classic_route_with_methods_kwarg() -> None:
    fr = _parse(
        """
        from flask import Flask
        app = Flask(__name__)

        @app.route('/health', methods=['GET', 'POST'])
        def health():
            return 'ok'
        """
    )
    method_paths = {(r.method, r.path) for r in fr.routes}
    assert ("GET", "/health") in method_paths
    assert ("POST", "/health") in method_paths


def test_requests_and_httpx_clients_captured() -> None:
    fr = _parse(
        """
        import requests, httpx

        def call_external():
            requests.get('https://api.example.com/users/42')
            httpx.post('/orders')
        """
    )
    by_method_path = {(h.method, h.path_template) for h in fr.http_clients}
    assert ("GET", "/users/42") in by_method_path
    assert ("POST", "/orders") in by_method_path
    bases = {h.base_url_hint for h in fr.http_clients}
    assert "https://api.example.com" in bases


def test_path_normalisation_handles_framework_syntaxes() -> None:
    assert normalize_path("/users/{id}") == "/users/{}"
    assert normalize_path("/users/{id:int}") == "/users/{}"
    assert normalize_path("/users/:id") == "/users/{}"
    assert normalize_path("/users/[id]") == "/users/{}"
    assert normalize_path("/users/<int:id>") == "/users/{}"
    assert normalize_path("users") == "/users"


def test_path_similarity_perfect_and_disjoint() -> None:
    assert path_similarity("/users/{}", "/users/{}") == 1.0
    assert path_similarity("/users/{}", "/orders/{}") == 0.5
    assert path_similarity("/users", "/users/{}") == 0.0  # different segment counts
