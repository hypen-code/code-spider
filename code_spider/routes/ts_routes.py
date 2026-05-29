"""TypeScript / JavaScript REST route + HTTP client extractors.

Targets (MVP):
    - **Express**             — ``app.get('/x', handler)``, ``router.post('/x', h)``
    - **NestJS**              — ``@Controller('prefix')`` class with
                                ``@Get('subpath')`` / ``@Post(...)`` methods
    - **HTTP clients**        — ``fetch('/url')``, template literals,
                                ``axios.get('/url')``, ``axios.post(...)``,
                                ``axios({url, method})`` (best effort)

Next.js filesystem-based routing (``pages/api/...``, ``app/.../route.ts``)
is detected via the file path, not the AST — see
:func:`infer_nextjs_route_from_path` for the helper used by the indexer.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from tree_sitter import Node

from code_spider.routes._common import is_http_method, normalize_method, normalize_path
from code_spider.symbols.fqn import qualify
from code_spider.symbols.model import HttpClientCall, Route, Span

_EXPRESS_METHODS: frozenset[str] = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "all"}
)
_NEST_METHOD_DECORATORS: frozenset[str] = frozenset(
    {"Get", "Post", "Put", "Patch", "Delete", "Head", "Options", "All"}
)
_AXIOS_OBJ_NAMES: frozenset[str] = frozenset({"axios"})
_HTTP_CLIENT_VAR_HINTS: frozenset[str] = frozenset(
    {"http", "api", "client", "request", "instance"}
)

_NEXTJS_DYNAMIC_SEGMENT = re.compile(r"\[\.\.\.([^/\[\]]+)\]|\[([^/\[\]]+)\]")


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _span(node: Node) -> Span:
    sr, sc = node.start_point
    er, ec = node.end_point
    return Span(start_line=sr + 1, start_col=sc, end_line=er + 1, end_col=ec)


def _string_value(node: Node, source: bytes) -> str | None:
    if node.type == "string":
        raw = _text(node, source)
        if len(raw) >= 2 and raw[0] in ("'", '"', "`") and raw[-1] == raw[0]:
            return raw[1:-1]
        return raw
    if node.type == "template_string":
        # ``/api/users/${id}`` -> normalise the ${...} parts to {} placeholders.
        raw = _text(node, source)
        inner = raw[1:-1] if raw.startswith("`") and raw.endswith("`") else raw
        return re.sub(r"\$\{[^}]+\}", "{}", inner)
    return None


def _attr_chain(node: Node, source: bytes) -> list[str]:
    parts: list[str] = []
    current: Node | None = node
    while current is not None:
        if current.type in {"identifier", "property_identifier"}:
            parts.append(_text(current, source))
            break
        if current.type == "member_expression":
            prop = current.child_by_field_name("property")
            obj = current.child_by_field_name("object")
            if prop is not None:
                parts.append(_text(prop, source))
            current = obj
            continue
        break
    parts.reverse()
    return parts


def _function_owner_name(call_node: Node, source: bytes) -> str | None:
    """Return the trailing identifier on the call's function chain, if any."""
    func = call_node.child_by_field_name("function")
    if func is None:
        return None
    chain = _attr_chain(func, source)
    return chain[-1] if chain else None


# ---------------------------------------------------------------- Routes


def extract_ts_routes(
    *, file_path: str, source: bytes, root: Node, module_fqn: str
) -> list[Route]:
    routes: list[Route] = []
    _walk_for_routes(
        node=root,
        source=source,
        file_path=file_path,
        module_fqn=module_fqn,
        out=routes,
        parent_fqn=module_fqn,
        controller_prefix="",
    )
    # Next.js file-based route (App Router & Pages API): if the file path
    # matches the convention, infer routes from any exported HTTP-named symbol.
    routes.extend(_infer_nextjs_routes_from_ast(file_path, source, root, module_fqn))
    return routes


def _walk_for_routes(
    *,
    node: Node,
    source: bytes,
    file_path: str,
    module_fqn: str,
    out: list[Route],
    parent_fqn: str,
    controller_prefix: str,
) -> None:
    # First scan this container for decorated classes (decorators are
    # siblings preceding the class_declaration in TS grammar).
    for decorators, target in _iter_decorated_targets(node):
        if target.type in {"class_declaration", "abstract_class_declaration"}:
            name_node = target.child_by_field_name("name")
            class_name = _text(name_node, source) if name_node else ""
            class_fqn = qualify(parent_fqn, class_name) if class_name else parent_fqn
            prefix = _nest_prefix_from_decorators(decorators, source) or controller_prefix
            body = target.child_by_field_name("body")
            if body is None:
                continue
            # Inside a class body, decorators precede each method_definition.
            for method_decs, method in _iter_decorated_targets(body):
                if method.type != "method_definition":
                    continue
                _maybe_nest_method_route(
                    method,
                    method_decs,
                    source,
                    file_path,
                    prefix,
                    class_fqn,
                    out,
                )

    # Then recurse into every named child to catch Express calls and nested scopes.
    for child in node.named_children:
        if child.type == "call_expression":
            _maybe_express_route(child, source, file_path, out)
        elif child.type in {"class_declaration", "abstract_class_declaration"}:
            # Already handled above; do not double-recurse into its body for routes.
            continue
        else:
            _walk_for_routes(
                node=child,
                source=source,
                file_path=file_path,
                module_fqn=module_fqn,
                out=out,
                parent_fqn=parent_fqn,
                controller_prefix=controller_prefix,
            )


def _iter_decorated_targets(container: Node):
    """Yield ``(decorators, target)`` for every direct child of ``container``.

    Decorators in TS grammar are *siblings* preceding their target. We walk
    the container's named children and accumulate consecutive decorators until
    a non-decorator node arrives; that node becomes the decorated target.
    Targets without any preceding decorator yield with an empty list.
    """
    decorators: list[Node] = []
    for child in container.named_children:
        if child.type == "decorator":
            decorators.append(child)
            continue
        yield decorators, child
        decorators = []


def _maybe_express_route(
    call: Node, source: bytes, file_path: str, out: list[Route]
) -> None:
    func = call.child_by_field_name("function")
    args = call.child_by_field_name("arguments")
    if func is None or args is None or func.type != "member_expression":
        return
    method = _function_owner_name(call, source)
    if method is None or method.lower() not in _EXPRESS_METHODS:
        return

    first = next(iter(args.named_children), None)
    if first is None:
        return
    path = _string_value(first, source)
    if path is None:
        return

    method_norm = "*" if method.lower() == "all" else normalize_method(method)
    # The handler is the next positional argument (usually inline arrow or
    # ref). For an inline handler we use the route's own coordinates as the
    # handler FQN (resolver may upgrade). For a reference, use that name.
    inline_anchor = f"<inline>@{file_path}:{call.start_point[0] + 1}"
    handler_fqn = _express_handler_fqn(args, source) or inline_anchor
    out.append(
        Route(
            method=method_norm,
            path=normalize_path(path),
            framework="express",
            handler_fqn=handler_fqn,
            file_path=file_path,
            span=_span(call),
        )
    )


def _express_handler_fqn(args: Node, source: bytes) -> str | None:
    """Inspect the 2nd positional argument and return a handler identifier."""
    pos: list[Node] = []
    for arg in args.named_children:
        if arg.type == "comment":
            continue
        pos.append(arg)
    if len(pos) < 2:
        return None
    handler = pos[1]
    if handler.type == "identifier":
        return _text(handler, source)
    if handler.type == "member_expression":
        chain = _attr_chain(handler, source)
        return ".".join(chain) if chain else None
    return None


def _nest_prefix_from_decorators(
    decorators: list[Node], source: bytes
) -> str | None:
    """Return the ``@Controller('prefix')`` prefix from a list of decorators."""
    for dec in decorators:
        prefix = _nest_decorator_string_arg(dec, source, "Controller")
        if prefix is not None:
            return normalize_path(prefix)
    return None


def _maybe_nest_method_route(
    method_node: Node,
    decorators: list[Node],
    source: bytes,
    file_path: str,
    controller_prefix: str,
    class_fqn: str,
    out: list[Route],
) -> None:
    name_node = method_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node, source)
    handler_fqn = qualify(class_fqn, method_name)
    for dec_name in _NEST_METHOD_DECORATORS:
        for dec in decorators:
            sub = _nest_decorator_string_arg(dec, source, dec_name)
            if sub is None:
                continue
            method_norm = "*" if dec_name == "All" else normalize_method(dec_name)
            full_path = (
                controller_prefix.rstrip("/") + "/" + sub.lstrip("/")
                if sub
                else controller_prefix or "/"
            )
            out.append(
                Route(
                    method=method_norm,
                    path=normalize_path(full_path),
                    framework="nestjs",
                    handler_fqn=handler_fqn,
                    file_path=file_path,
                    span=_span(dec),
                )
            )


def _nest_decorator_string_arg(
    decorator: Node, source: bytes, expected_name: str
) -> str | None:
    """If decorator is ``@Name(...)``, return the first string arg (or '' for ``@Name()``)."""
    if not decorator.named_children:
        return None
    expr = decorator.named_children[0]
    if expr.type == "identifier":
        return "" if _text(expr, source) == expected_name else None
    if expr.type != "call_expression":
        return None
    func = expr.child_by_field_name("function")
    if func is None or func.type != "identifier":
        return None
    if _text(func, source) != expected_name:
        return None
    args = expr.child_by_field_name("arguments")
    if args is None:
        return ""
    for arg in args.named_children:
        if arg.type in {"string", "template_string"}:
            return _string_value(arg, source) or ""
    return ""


# ---------------------------------------------------------- Next.js inference


def _infer_nextjs_routes_from_ast(
    file_path: str, source: bytes, root: Node, module_fqn: str
) -> list[Route]:
    """Detect Next.js file-based routes from the file path + exported handlers."""
    p = PurePosixPath(file_path)
    parts = list(p.parts)

    # App Router: app/.../route.{ts,js,tsx,jsx}
    if (
        "app" in parts
        and p.stem == "route"
        and parts.index("app") < len(parts) - 1
    ):
        route_path = _nextjs_app_router_path(parts)
        return _emit_nextjs_handlers(
            file_path=file_path,
            source=source,
            root=root,
            route_path=route_path,
            module_fqn=module_fqn,
            framework="nextjs-app",
        )

    # Pages API: pages/api/...
    if "pages" in parts and "api" in parts:
        idx = parts.index("pages")
        if idx + 1 < len(parts) and parts[idx + 1] == "api":
            route_path = _nextjs_pages_path(parts[idx + 1 :], p.stem)
            return _emit_nextjs_handlers(
                file_path=file_path,
                source=source,
                root=root,
                route_path=route_path,
                module_fqn=module_fqn,
                framework="nextjs-pages",
            )

    return []


def _nextjs_app_router_path(parts: list[str]) -> str:
    app_idx = parts.index("app")
    segments = parts[app_idx + 1 : -1]  # drop the trailing "route.ts"
    return "/" + "/".join(_NEXTJS_DYNAMIC_SEGMENT.sub("{}", s) for s in segments)


def _nextjs_pages_path(api_parts: list[str], stem: str) -> str:
    # api_parts[0] == "api"
    rest = list(api_parts[1:])
    file_segment = stem if stem != "index" else ""
    if file_segment:
        rest.append(file_segment)
    return "/api/" + "/".join(_NEXTJS_DYNAMIC_SEGMENT.sub("{}", s) for s in rest if s)


def _emit_nextjs_handlers(
    *,
    file_path: str,
    source: bytes,
    root: Node,
    route_path: str,
    module_fqn: str,
    framework: str,
) -> list[Route]:
    out: list[Route] = []
    span = _span(root)
    # App Router: look for ``export async function GET/POST/...``
    if framework == "nextjs-app":
        for child in root.named_children:
            if child.type != "export_statement":
                continue
            decl = next(
                (
                    c
                    for c in child.named_children
                    if c.type in {"function_declaration", "lexical_declaration"}
                ),
                None,
            )
            if decl is None:
                continue
            name_node = decl.child_by_field_name("name")
            if name_node is None and decl.type == "lexical_declaration":
                for sub in decl.named_children:
                    if sub.type == "variable_declarator":
                        name_node = sub.child_by_field_name("name")
                        break
            if name_node is None:
                continue
            name = _text(name_node, source)
            if not is_http_method(name):
                continue
            out.append(
                Route(
                    method=normalize_method(name),
                    path=normalize_path(route_path),
                    framework=framework,
                    handler_fqn=qualify(module_fqn, name),
                    file_path=file_path,
                    span=span,
                )
            )
        return out

    # Pages API: a single default export handler responds to all methods.
    out.append(
        Route(
            method="*",
            path=normalize_path(route_path),
            framework=framework,
            handler_fqn=qualify(module_fqn, "default"),
            file_path=file_path,
            span=span,
        )
    )
    return out


# --------------------------------------------------------- HTTP clients (TS/JS)


def extract_ts_http_clients(
    *, file_path: str, source: bytes, root: Node, module_fqn: str
) -> list[HttpClientCall]:
    out: list[HttpClientCall] = []
    _walk_for_clients(root, source, file_path, out, caller_fqn=module_fqn)
    return out


def _walk_for_clients(
    node: Node,
    source: bytes,
    file_path: str,
    out: list[HttpClientCall],
    caller_fqn: str,
) -> None:
    for child in node.named_children:
        if child.type in {
            "function_declaration",
            "method_definition",
            "class_declaration",
            "abstract_class_declaration",
        }:
            name_node = child.child_by_field_name("name")
            inner = (
                qualify(caller_fqn, _text(name_node, source))
                if name_node is not None
                else caller_fqn
            )
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_for_clients(body, source, file_path, out, inner)
            continue

        if child.type in {"lexical_declaration", "variable_declaration"}:
            for decl in child.named_children:
                if decl.type != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                name_node = decl.child_by_field_name("name")
                if (
                    value is not None
                    and value.type in {"arrow_function", "function_expression"}
                    and name_node is not None
                    and name_node.type == "identifier"
                ):
                    inner = qualify(caller_fqn, _text(name_node, source))
                    body = value.child_by_field_name("body")
                    if body is not None:
                        _walk_for_clients(body, source, file_path, out, inner)
                    continue
                _walk_for_clients(decl, source, file_path, out, caller_fqn)
            continue

        if child.type == "call_expression":
            hit = _maybe_http_call(child, source, file_path, caller_fqn)
            if hit is not None:
                out.append(hit)

        _walk_for_clients(child, source, file_path, out, caller_fqn)


def _maybe_http_call(
    call: Node, source: bytes, file_path: str, caller_fqn: str
) -> HttpClientCall | None:
    func = call.child_by_field_name("function")
    args = call.child_by_field_name("arguments")
    if func is None or args is None:
        return None

    # Bare `fetch('/url', { method: 'POST' })`.
    if func.type == "identifier" and _text(func, source) == "fetch":
        return _fetch_call(call, args, source, file_path, caller_fqn)

    if func.type != "member_expression":
        return None

    chain = _attr_chain(func, source)
    if len(chain) < 2:
        return None
    method = chain[-1].lower()
    target = chain[-2]

    if not is_http_method(method):
        return None
    if (
        target not in _AXIOS_OBJ_NAMES
        and target.lower() not in _HTTP_CLIENT_VAR_HINTS
    ):
        return None

    first = next(iter(args.named_children), None)
    if first is None:
        return None
    url = _string_value(first, source)
    if url is None:
        return None

    base, path = _split_url(url)
    return HttpClientCall(
        caller_fqn=caller_fqn,
        method=normalize_method(method),
        path_template=normalize_path(path),
        base_url_hint=base,
        file_path=file_path,
        span=_span(call),
    )


def _fetch_call(
    call: Node,
    args: Node,
    source: bytes,
    file_path: str,
    caller_fqn: str,
) -> HttpClientCall | None:
    """Detect a bare ``fetch(url, opts?)`` invocation."""
    positional: list[Node] = [a for a in args.named_children if a.type != "comment"]
    if not positional:
        return None
    url_node = positional[0]
    url = _string_value(url_node, source)
    if url is None:
        return None
    method = "GET"
    if len(positional) >= 2 and positional[1].type == "object":
        method = _options_method(positional[1], source) or "GET"

    base, path = _split_url(url)
    return HttpClientCall(
        caller_fqn=caller_fqn,
        method=normalize_method(method),
        path_template=normalize_path(path),
        base_url_hint=base,
        file_path=file_path,
        span=_span(call),
    )


def _options_method(obj_node: Node, source: bytes) -> str | None:
    """Find ``{ method: 'POST', ... }`` in a fetch options literal."""
    for pair in obj_node.named_children:
        if pair.type != "pair":
            continue
        key = pair.child_by_field_name("key")
        value = pair.child_by_field_name("value")
        if key is None or value is None:
            continue
        key_text = _text(key, source).strip("'\"")
        if key_text != "method":
            continue
        v = _string_value(value, source)
        if v and is_http_method(v):
            return v.upper()
    return None


def _split_url(url: str) -> tuple[str | None, str]:
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if "/" in rest:
            host, path = rest.split("/", 1)
            return f"{scheme}://{host}", "/" + path
        return f"{scheme}://{rest}", "/"
    return None, url
