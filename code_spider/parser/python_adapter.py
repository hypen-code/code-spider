"""Python language adapter — Tree-sitter parser + symbol extractor.

Phase 0 scope:
    - Parse Python source with tree-sitter-python (error-tolerant).
    - Extract :class:`Symbol` records for top-level + nested functions, classes,
      and methods. FQNs are computed from the module path (``code_spider.parser
      .python_adapter``) joined with class/function nesting.
    - Capture docstrings (first string literal in a function/class body).
    - Capture imports as :class:`Import` records (resolved_fqn left empty;
      the Phase 1 resolver fills it).
    - Capture call sites as :class:`CallSite` records (also resolved later).

Out of scope here: routes, kafka extractors, chunking — those live in their
own modules to keep parsing fast and per-feature testable.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from code_spider.messaging.python_kafka import extract_python_kafka
from code_spider.routes.python_routes import (
    extract_python_http_clients,
    extract_python_routes,
)
from code_spider.symbols.fqn import file_to_module_fqn, qualify
from code_spider.symbols.model import (
    CallSite,
    FileRecord,
    Import,
    Module,
    Span,
    Symbol,
    SymbolKind,
)

_PY_LANGUAGE: Final = Language(tspython.language())


def _span(node: Node) -> Span:
    """Tree-sitter points are (row, col), 0-indexed. We expose 1-indexed lines."""
    sr, sc = node.start_point
    er, ec = node.end_point
    return Span(start_line=sr + 1, start_col=sc, end_line=er + 1, end_col=ec)


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _signature(def_node: Node, source: bytes) -> str:
    """First line of a function/class definition (the ``def foo(...)``/``class Foo(...)``)."""
    body = def_node.child_by_field_name("body")
    end_byte = body.start_byte if body is not None else def_node.end_byte
    raw = source[def_node.start_byte : end_byte].decode("utf-8", errors="replace")
    # Collapse multi-line signatures into a single readable line.
    cleaned = " ".join(part.strip() for part in raw.splitlines() if part.strip())
    return cleaned.rstrip(":").strip()


def _docstring(def_node: Node, source: bytes) -> str:
    """First string-literal expression statement inside a function/class body, or ''."""
    body = def_node.child_by_field_name("body")
    if body is None:
        return ""
    for child in body.named_children:
        if child.type == "expression_statement" and child.named_child_count == 1:
            inner = child.named_children[0]
            if inner.type == "string":
                raw = _text(inner, source)
                # Strip the surrounding quotes (handles ''', \"\"\", '', \"\").
                return _strip_python_string_quotes(raw)
            return ""
        return ""
    return ""


def _strip_python_string_quotes(raw: str) -> str:
    """Remove leading/trailing Python string quotes (incl. triple-quoted) + prefixes."""
    s = raw.lstrip()
    # Drop string prefixes like r, b, u, rb, br, f, etc.
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    s = s[i:]
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q):
            return s[len(q) : -len(q)].strip()
    return s.strip()


def _visibility(name: str) -> str:
    # Dunder names (``__init__``, ``__repr__``, etc.) are public.
    if name.startswith("__") and name.endswith("__"):
        return "public"
    if name.startswith("__"):
        return "private"
    if name.startswith("_"):
        return "protected"
    return "public"


class PythonAdapter:
    """Tree-sitter-backed Python parser. See module docstring for scope."""

    lang: str = "python"
    extensions: tuple[str, ...] = (".py", ".pyi")

    def __init__(self) -> None:
        self._parser = Parser(_PY_LANGUAGE)

    def parse_file(self, repo_relative_path: str, source: bytes) -> FileRecord:
        tree = self._parser.parse(source)
        root = tree.root_node

        module_fqn = file_to_module_fqn(repo_relative_path, self.lang)
        module = Module(
            fqn=module_fqn,
            kind="package" if repo_relative_path.endswith("__init__.py") else "module",
            file_path=repo_relative_path,
        )

        symbols: list[Symbol] = []
        imports: list[Import] = []
        call_sites: list[CallSite] = []

        # Walk the AST recursively, tracking the enclosing FQN for nested defs.
        self._walk(
            node=root,
            source=source,
            file_path=repo_relative_path,
            parent_fqn=module_fqn,
            parent_kind_is_class=False,
            symbols=symbols,
            imports=imports,
            call_sites=call_sites,
        )

        routes = extract_python_routes(
            file_path=repo_relative_path,
            source=source,
            root=root,
            module_fqn=module_fqn,
        )
        http_clients = extract_python_http_clients(
            file_path=repo_relative_path,
            source=source,
            root=root,
            module_fqn=module_fqn,
        )
        kafka_producers, kafka_consumers = extract_python_kafka(
            file_path=repo_relative_path,
            source=source,
            root=root,
            module_fqn=module_fqn,
        )

        return FileRecord(
            repo_relative_path=repo_relative_path,
            lang=self.lang,
            hash_blake3="",  # filled by the indexer pipeline (file walk stage)
            size_bytes=len(source),
            line_count=source.count(b"\n") + 1,
            module=module,
            symbols=symbols,
            imports=imports,
            call_sites=call_sites,
            routes=routes,
            http_clients=http_clients,
            kafka_producers=kafka_producers,
            kafka_consumers=kafka_consumers,
        )

    # ------------------------------------------------------------------ walk

    def _walk(
        self,
        *,
        node: Node,
        source: bytes,
        file_path: str,
        parent_fqn: str,
        parent_kind_is_class: bool,
        symbols: list[Symbol],
        imports: list[Import],
        call_sites: list[CallSite],
    ) -> None:
        for child in node.named_children:
            t = child.type

            if t in {"function_definition", "class_definition"}:
                self._emit_definition(
                    def_node=child,
                    source=source,
                    file_path=file_path,
                    parent_fqn=parent_fqn,
                    parent_kind_is_class=parent_kind_is_class,
                    symbols=symbols,
                    imports=imports,
                    call_sites=call_sites,
                )

            elif t == "decorated_definition":
                inner = child.child_by_field_name("definition")
                if inner is not None and inner.type in {
                    "function_definition",
                    "class_definition",
                }:
                    self._emit_definition(
                        def_node=inner,
                        source=source,
                        file_path=file_path,
                        parent_fqn=parent_fqn,
                        parent_kind_is_class=parent_kind_is_class,
                        symbols=symbols,
                        imports=imports,
                        call_sites=call_sites,
                    )

            elif t in {"import_statement", "import_from_statement"}:
                imports.extend(self._extract_imports(child, source, file_path))

            elif t == "call":
                call_sites.append(self._extract_call(child, source, file_path, parent_fqn))
                # Recurse to catch nested calls inside args.
                self._walk(
                    node=child,
                    source=source,
                    file_path=file_path,
                    parent_fqn=parent_fqn,
                    parent_kind_is_class=False,
                    symbols=symbols,
                    imports=imports,
                    call_sites=call_sites,
                )

            else:
                # Descend into compound statements (if/for/with/try/etc.) so we
                # see imports and calls inside conditional branches.
                self._walk(
                    node=child,
                    source=source,
                    file_path=file_path,
                    parent_fqn=parent_fqn,
                    parent_kind_is_class=parent_kind_is_class,
                    symbols=symbols,
                    imports=imports,
                    call_sites=call_sites,
                )

    # --------------------------------------------------------------- emitters

    def _emit_definition(
        self,
        *,
        def_node: Node,
        source: bytes,
        file_path: str,
        parent_fqn: str,
        parent_kind_is_class: bool,
        symbols: list[Symbol],
        imports: list[Import],
        call_sites: list[CallSite],
    ) -> None:
        name_node = def_node.child_by_field_name("name")
        if name_node is None:
            return
        name = _text(name_node, source)
        fqn = qualify(parent_fqn, name)

        if def_node.type == "class_definition":
            kind = SymbolKind.CLASS
            child_is_class = True
        else:
            kind = SymbolKind.METHOD if parent_kind_is_class else SymbolKind.FUNCTION
            child_is_class = False

        symbols.append(
            Symbol(
                fqn=fqn,
                name=name,
                kind=kind,
                lang=self.lang,
                file_path=file_path,
                span=_span(def_node),
                signature=_signature(def_node, source),
                docstring=_docstring(def_node, source),
                visibility=_visibility(name),
                parent_fqn=parent_fqn,
            )
        )

        body = def_node.child_by_field_name("body")
        if body is not None:
            self._walk(
                node=body,
                source=source,
                file_path=file_path,
                parent_fqn=fqn,
                parent_kind_is_class=child_is_class,
                symbols=symbols,
                imports=imports,
                call_sites=call_sites,
            )

    def _extract_imports(
        self, node: Node, source: bytes, file_path: str
    ) -> Iterator[Import]:
        raw = _text(node, source).strip()
        span = _span(node)
        if node.type == "import_statement":
            for child in node.named_children:
                if child.type == "dotted_name":
                    target = _text(child, source).strip()
                    yield Import(
                        file_path=file_path,
                        raw=raw,
                        local_name=target.split(".")[0],
                        target_fqn=target,
                        span=span,
                    )
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    alias_node = child.child_by_field_name("alias")
                    if name_node is None or alias_node is None:
                        continue
                    yield Import(
                        file_path=file_path,
                        raw=raw,
                        local_name=_text(alias_node, source).strip(),
                        target_fqn=_text(name_node, source).strip(),
                        span=span,
                    )
        else:  # import_from_statement
            module_node = node.child_by_field_name("module_name")
            module_fqn = _text(module_node, source).strip() if module_node else ""
            for child in node.named_children:
                if child is module_node:
                    continue
                if child.type == "dotted_name":
                    name = _text(child, source).strip()
                    target = f"{module_fqn}.{name}" if module_fqn else name
                    yield Import(
                        file_path=file_path,
                        raw=raw,
                        local_name=name.split(".")[-1],
                        target_fqn=target,
                        span=span,
                    )
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    alias_node = child.child_by_field_name("alias")
                    if name_node is None or alias_node is None:
                        continue
                    name = _text(name_node, source).strip()
                    target = f"{module_fqn}.{name}" if module_fqn else name
                    yield Import(
                        file_path=file_path,
                        raw=raw,
                        local_name=_text(alias_node, source).strip(),
                        target_fqn=target,
                        span=span,
                    )
                elif child.type == "wildcard_import":
                    yield Import(
                        file_path=file_path,
                        raw=raw,
                        local_name="*",
                        target_fqn=f"{module_fqn}.*" if module_fqn else "*",
                        span=span,
                    )

    def _extract_call(
        self, node: Node, source: bytes, file_path: str, caller_fqn: str
    ) -> CallSite:
        func = node.child_by_field_name("function")
        call_text = _text(func, source).strip() if func is not None else _text(node, source).strip()
        return CallSite(
            caller_fqn=caller_fqn,
            file_path=file_path,
            call_text=call_text,
            span=_span(node),
        )
