"""Python REST route + HTTP client extractors.

Targets (MVP):
    - **FastAPI** / APIRouter — ``@app.get('/x')``, ``@router.post('/x')``
    - **Flask**               — ``@app.route('/x', methods=[...])``,
                                ``@app.get('/x')`` (Flask 2.x convenience)
    - **HTTP clients**        — ``requests.{get,post,...}``, ``httpx.{get,...}``,
                                ``client.{get,...}`` on Session/AsyncClient

Each extractor is a pure function that walks the Tree-sitter AST and produces
:class:`Route` / :class:`HttpClientCall` records. They are invoked from
:mod:`code_spider.parser.python_adapter` during file parsing.
"""

from __future__ import annotations

from tree_sitter import Node

from code_spider.routes._common import is_http_method, normalize_method, normalize_path
from code_spider.symbols.fqn import qualify
from code_spider.symbols.model import HttpClientCall, Route, Span

# Decorator attribute names that signal a route definition (FastAPI/Flask).
_ROUTE_METHOD_ATTRS: frozenset[str] = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options"}
)

# HTTP client modules / variable hints — heuristic detection.
_HTTP_CLIENT_MODULES: frozenset[str] = frozenset({"requests", "httpx"})

# Common variable names for HTTP client instances.
_CLIENT_VAR_HINTS: frozenset[str] = frozenset(
    {"client", "session", "http", "async_client", "http_client"}
)


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _span(node: Node) -> Span:
    sr, sc = node.start_point
    er, ec = node.end_point
    return Span(start_line=sr + 1, start_col=sc, end_line=er + 1, end_col=ec)


def _string_literal_value(node: Node, source: bytes) -> str | None:
    """Return the contents of a string literal node, or None if not a literal."""
    if node.type != "string":
        return None
    raw = _text(node, source)
    # Strip prefixes like b, r, f, u (and combinations).
    s = raw
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    s = s[i:]
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q):
            return s[len(q) : -len(q)]
    return None


def _attribute_chain(node: Node, source: bytes) -> list[str]:
    """For an ``attribute`` or ``identifier`` node, return its dotted chain.

    e.g. ``requests.get`` -> ``["requests", "get"]``;
         ``self.client.get`` -> ``["self", "client", "get"]``;
         ``foo()`` -> ``["foo"]``.
    """
    parts: list[str] = []
    current: Node | None = node
    while current is not None:
        if current.type == "identifier":
            parts.append(_text(current, source))
            break
        if current.type == "attribute":
            attr = current.child_by_field_name("attribute")
            obj = current.child_by_field_name("object")
            if attr is not None:
                parts.append(_text(attr, source))
            current = obj
            continue
        break
    parts.reverse()
    return parts


# --------------------------------------------------------------- Route extract


def extract_python_routes(
    *, file_path: str, source: bytes, root: Node, module_fqn: str
) -> list[Route]:
    """Find FastAPI / Flask route decorators and link them to handler symbols."""
    routes: list[Route] = []
    _walk_for_routes(root, source, file_path, module_fqn, routes, parent_fqn=module_fqn)
    return routes


def _walk_for_routes(
    node: Node,
    source: bytes,
    file_path: str,
    module_fqn: str,
    out: list[Route],
    parent_fqn: str,
) -> None:
    for child in node.named_children:
        if child.type == "decorated_definition":
            inner = child.child_by_field_name("definition")
            if inner is None or inner.type != "function_definition":
                _walk_for_routes(child, source, file_path, module_fqn, out, parent_fqn)
                continue
            name_node = inner.child_by_field_name("name")
            if name_node is None:
                continue
            handler_name = _text(name_node, source)
            handler_fqn = qualify(parent_fqn, handler_name)
            for dec in _decorators(child):
                emitted = _route_from_decorator(
                    dec, source, file_path, handler_fqn
                )
                out.extend(emitted)
            # Recurse into the function body so nested handlers (rare) are found.
            body = inner.child_by_field_name("body")
            if body is not None:
                _walk_for_routes(body, source, file_path, module_fqn, out, handler_fqn)

        elif child.type in {"function_definition", "class_definition"}:
            name_node = child.child_by_field_name("name")
            name = _text(name_node, source) if name_node is not None else ""
            inner_parent = qualify(parent_fqn, name) if name else parent_fqn
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_for_routes(
                    body, source, file_path, module_fqn, out, inner_parent
                )

        else:
            _walk_for_routes(child, source, file_path, module_fqn, out, parent_fqn)


def _decorators(decorated: Node) -> list[Node]:
    return [c for c in decorated.named_children if c.type == "decorator"]


def _route_from_decorator(
    decorator: Node, source: bytes, file_path: str, handler_fqn: str
) -> list[Route]:
    """Parse a decorator like ``@app.get('/x')`` into Route records."""
    # decorator.named_children[0] is the decorator expression.
    if not decorator.named_children:
        return []
    expr = decorator.named_children[0]
    call_node: Node | None
    if expr.type == "call":
        call_node = expr
    else:
        return []

    func = call_node.child_by_field_name("function")
    args = call_node.child_by_field_name("arguments")
    if func is None or args is None or func.type != "attribute":
        return []

    chain = _attribute_chain(func, source)
    if len(chain) < 2:
        return []
    attr = chain[-1]
    method_lower = attr.lower()

    # FastAPI / Flask 2.x: @app.get('/x') / @router.post('/x').
    if method_lower in _ROUTE_METHOD_ATTRS:
        path = _first_string_arg(args, source)
        if path is None:
            return []
        return [
            Route(
                method=normalize_method(method_lower),
                path=normalize_path(path),
                framework="fastapi-or-flask",
                handler_fqn=handler_fqn,
                file_path=file_path,
                span=_span(decorator),
            )
        ]

    # Flask classic: @app.route('/x', methods=['GET','POST']).
    if attr == "route":
        path = _first_string_arg(args, source)
        if path is None:
            return []
        methods = _flask_methods_kwarg(args, source) or ["GET"]
        return [
            Route(
                method=normalize_method(m),
                path=normalize_path(path),
                framework="flask",
                handler_fqn=handler_fqn,
                file_path=file_path,
                span=_span(decorator),
            )
            for m in methods
        ]

    return []


def _first_string_arg(arguments: Node, source: bytes) -> str | None:
    for arg in arguments.named_children:
        if arg.type == "string":
            return _string_literal_value(arg, source)
        if arg.type == "keyword_argument":
            continue
        return None  # first positional arg was not a string literal
    return None


def _flask_methods_kwarg(arguments: Node, source: bytes) -> list[str] | None:
    """Parse ``methods=['GET','POST']`` from a Flask ``@app.route`` decorator."""
    for arg in arguments.named_children:
        if arg.type != "keyword_argument":
            continue
        name_node = arg.child_by_field_name("name")
        value_node = arg.child_by_field_name("value")
        if name_node is None or value_node is None:
            continue
        if _text(name_node, source) != "methods":
            continue
        if value_node.type not in {"list", "tuple"}:
            continue
        methods: list[str] = []
        for elem in value_node.named_children:
            val = _string_literal_value(elem, source)
            if val and is_http_method(val):
                methods.append(val.upper())
        return methods or None
    return None


# ---------------------------------------------------------- HTTP client extract


def extract_python_http_clients(
    *, file_path: str, source: bytes, root: Node, module_fqn: str
) -> list[HttpClientCall]:
    """Detect requests/httpx-style HTTP client calls and tag each with its caller."""
    out: list[HttpClientCall] = []
    _walk_for_http_clients(root, source, file_path, out, caller_fqn=module_fqn)
    return out


def _walk_for_http_clients(
    node: Node,
    source: bytes,
    file_path: str,
    out: list[HttpClientCall],
    caller_fqn: str,
) -> None:
    for child in node.named_children:
        if child.type in {"function_definition", "class_definition"}:
            name_node = child.child_by_field_name("name")
            inner = qualify(caller_fqn, _text(name_node, source)) if name_node else caller_fqn
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_for_http_clients(body, source, file_path, out, inner)
            continue

        if child.type == "decorated_definition":
            inner_def = child.child_by_field_name("definition")
            if inner_def is not None and inner_def.type in {
                "function_definition",
                "class_definition",
            }:
                name_node = inner_def.child_by_field_name("name")
                inner = (
                    qualify(caller_fqn, _text(name_node, source))
                    if name_node
                    else caller_fqn
                )
                body = inner_def.child_by_field_name("body")
                if body is not None:
                    _walk_for_http_clients(body, source, file_path, out, inner)
                continue

        if child.type == "call":
            hit = _python_http_call_from(child, source, file_path, caller_fqn)
            if hit is not None:
                out.append(hit)
            # Still recurse — nested calls can contain HTTP calls in their args.

        _walk_for_http_clients(child, source, file_path, out, caller_fqn)


def _python_http_call_from(
    call: Node, source: bytes, file_path: str, caller_fqn: str
) -> HttpClientCall | None:
    func = call.child_by_field_name("function")
    if func is None or func.type != "attribute":
        return None
    chain = _attribute_chain(func, source)
    if len(chain) < 2:
        return None

    method_attr = chain[-1].lower()
    if not is_http_method(method_attr):
        return None

    target = chain[-2]
    # Heuristic: ``requests.get(...)`` / ``httpx.get(...)`` / ``client.get(...)``
    # / ``self.client.get(...)`` / ``async_client.get(...)``.
    if (
        target not in _HTTP_CLIENT_MODULES
        and target.lower() not in _CLIENT_VAR_HINTS
    ):
        return None

    args = call.child_by_field_name("arguments")
    if args is None:
        return None
    url_str = _first_string_arg(args, source)
    if url_str is None:
        return None

    base, path = _split_url(url_str)
    return HttpClientCall(
        caller_fqn=caller_fqn,
        method=normalize_method(method_attr),
        path_template=normalize_path(path),
        base_url_hint=base,
        file_path=file_path,
        span=_span(call),
    )


def _split_url(url: str) -> tuple[str | None, str]:
    """Split an absolute/relative URL into (scheme+host, path)."""
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if "/" in rest:
            host, path = rest.split("/", 1)
            return f"{scheme}://{host}", "/" + path
        return f"{scheme}://{rest}", "/"
    return None, url
