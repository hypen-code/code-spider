"""Shared Tree-sitter walker for TypeScript / JavaScript / JSX.

The TS and JS grammars are nearly identical at the level we care about
(declarations + imports + calls); only TS adds ``interface_declaration``,
``type_alias_declaration``, ``enum_declaration``, and decorator metadata.
This base class implements the common walker; subclasses provide the
:class:`tree_sitter.Language` instance and extensions.

JSDoc docstrings are captured when a ``comment`` node ending with ``*/``
appears immediately before a declaration.
"""

from __future__ import annotations

from collections.abc import Iterator

from tree_sitter import Language, Node, Parser

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

# Node types that represent a named, hoisted, definition. Body-bearing nodes
# get descended into so we capture nested class methods, inner functions, etc.
_DEFINITION_NODES: frozenset[str] = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "internal_module",  # TS namespace
    }
)


def _span(node: Node) -> Span:
    sr, sc = node.start_point
    er, ec = node.end_point
    return Span(start_line=sr + 1, start_col=sc, end_line=er + 1, end_col=ec)


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _string_literal_value(node: Node, source: bytes) -> str | None:
    if node.type != "string":
        return None
    raw = _text(node, source)
    for q in ("'", '"', "`"):
        if raw.startswith(q) and raw.endswith(q) and len(raw) >= 2:
            return raw[1:-1]
    return raw


def _strip_jsdoc(raw: str) -> str:
    """Strip ``/** ... */`` and leading ``*`` line markers; return clean text."""
    s = raw.strip()
    if not (s.startswith("/**") and s.endswith("*/")):
        return ""
    inner = s[3:-2]
    cleaned: list[str] = []
    for raw_line in inner.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("*"):
            stripped = stripped[1:].lstrip()
        cleaned.append(stripped)
    return "\n".join(line for line in cleaned if line).strip()


def _preceding_jsdoc(node: Node, source: bytes) -> str:
    """Return the JSDoc block immediately preceding ``node`` (or empty)."""
    prev = node.prev_named_sibling
    if prev is None or prev.type != "comment":
        return ""
    raw = _text(prev, source)
    return _strip_jsdoc(raw)


def _visibility(name: str) -> str:
    if name.startswith("_"):
        return "protected"
    return "public"


def _signature(def_node: Node, source: bytes) -> str:
    """Return the header line(s) of a declaration up to its body/block."""
    body = def_node.child_by_field_name("body")
    end_byte = body.start_byte if body is not None else def_node.end_byte
    raw = source[def_node.start_byte : end_byte].decode("utf-8", errors="replace")
    cleaned = " ".join(part.strip() for part in raw.splitlines() if part.strip())
    return cleaned.rstrip("{").strip()


def _kind_for(node_type: str, *, parent_is_class: bool) -> SymbolKind:
    if node_type in {"class_declaration", "abstract_class_declaration"}:
        return SymbolKind.CLASS
    if node_type == "interface_declaration":
        return SymbolKind.INTERFACE
    if node_type == "type_alias_declaration":
        return SymbolKind.TYPE_ALIAS
    if node_type == "enum_declaration":
        return SymbolKind.CLASS  # treat enums as classes for graph purposes
    if node_type == "method_definition":
        return SymbolKind.METHOD
    if node_type == "internal_module":
        return SymbolKind.CLASS  # namespace-as-container
    if parent_is_class:
        return SymbolKind.METHOD
    return SymbolKind.FUNCTION


class TsJsAdapterBase:
    """Common Tree-sitter walker for TypeScript and JavaScript sources."""

    lang: str = ""
    extensions: tuple[str, ...] = ()

    def __init__(self, ts_language: Language) -> None:
        self._parser = Parser(ts_language)

    def parse_file(self, repo_relative_path: str, source: bytes) -> FileRecord:
        tree = self._parser.parse(source)
        root = tree.root_node
        module_fqn = file_to_module_fqn(repo_relative_path, self.lang)

        module = Module(
            fqn=module_fqn,
            kind="module",
            file_path=repo_relative_path,
        )

        symbols: list[Symbol] = []
        imports: list[Import] = []
        call_sites: list[CallSite] = []

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

        return FileRecord(
            repo_relative_path=repo_relative_path,
            lang=self.lang,
            hash_blake3="",
            size_bytes=len(source),
            line_count=source.count(b"\n") + 1,
            module=module,
            symbols=symbols,
            imports=imports,
            call_sites=call_sites,
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

            if t in _DEFINITION_NODES:
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

            elif t in {"lexical_declaration", "variable_declaration"}:
                for decl in child.named_children:
                    if decl.type != "variable_declarator":
                        continue
                    self._emit_variable_decl(
                        decl_node=decl,
                        source=source,
                        file_path=file_path,
                        parent_fqn=parent_fqn,
                        parent_kind_is_class=parent_kind_is_class,
                        symbols=symbols,
                        imports=imports,
                        call_sites=call_sites,
                    )

            elif t == "public_field_definition":
                # TypeScript class field — may hold an arrow function.
                self._emit_class_field(
                    field_node=child,
                    source=source,
                    file_path=file_path,
                    parent_fqn=parent_fqn,
                    symbols=symbols,
                    imports=imports,
                    call_sites=call_sites,
                )

            elif t == "import_statement":
                imports.extend(self._extract_imports(child, source, file_path))

            elif t == "call_expression":
                call_sites.append(self._extract_call(child, source, file_path, parent_fqn))
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

        kind = _kind_for(def_node.type, parent_is_class=parent_kind_is_class)
        is_class_like = kind in {SymbolKind.CLASS, SymbolKind.INTERFACE}

        symbols.append(
            Symbol(
                fqn=fqn,
                name=name,
                kind=kind,
                lang=self.lang,
                file_path=file_path,
                span=_span(def_node),
                signature=_signature(def_node, source),
                docstring=_preceding_jsdoc(def_node, source),
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
                parent_kind_is_class=is_class_like,
                symbols=symbols,
                imports=imports,
                call_sites=call_sites,
            )

    def _emit_variable_decl(
        self,
        *,
        decl_node: Node,
        source: bytes,
        file_path: str,
        parent_fqn: str,
        parent_kind_is_class: bool,
        symbols: list[Symbol],
        imports: list[Import],
        call_sites: list[CallSite],
    ) -> None:
        name_node = decl_node.child_by_field_name("name")
        value_node = decl_node.child_by_field_name("value")
        if name_node is None or name_node.type != "identifier":
            return  # ignore destructuring for now
        name = _text(name_node, source)

        if value_node is not None and value_node.type in {
            "arrow_function",
            "function_expression",
            "generator_function",
        }:
            fqn = qualify(parent_fqn, name)
            symbols.append(
                Symbol(
                    fqn=fqn,
                    name=name,
                    kind=SymbolKind.METHOD
                    if parent_kind_is_class
                    else SymbolKind.FUNCTION,
                    lang=self.lang,
                    file_path=file_path,
                    span=_span(decl_node),
                    signature=_signature(decl_node, source),
                    docstring=_preceding_jsdoc(decl_node, source),
                    visibility=_visibility(name),
                    parent_fqn=parent_fqn,
                )
            )
            body = value_node.child_by_field_name("body")
            if body is not None:
                self._walk(
                    node=body,
                    source=source,
                    file_path=file_path,
                    parent_fqn=fqn,
                    parent_kind_is_class=False,
                    symbols=symbols,
                    imports=imports,
                    call_sites=call_sites,
                )
            return

        # Plain constant — emit as a Variable so it's reachable by FQN searches.
        symbols.append(
            Symbol(
                fqn=qualify(parent_fqn, name),
                name=name,
                kind=SymbolKind.VARIABLE,
                lang=self.lang,
                file_path=file_path,
                span=_span(decl_node),
                signature=_signature(decl_node, source),
                docstring=_preceding_jsdoc(decl_node, source),
                visibility=_visibility(name),
                parent_fqn=parent_fqn,
            )
        )
        if value_node is not None:
            self._walk(
                node=value_node,
                source=source,
                file_path=file_path,
                parent_fqn=parent_fqn,
                parent_kind_is_class=parent_kind_is_class,
                symbols=symbols,
                imports=imports,
                call_sites=call_sites,
            )

    def _emit_class_field(
        self,
        *,
        field_node: Node,
        source: bytes,
        file_path: str,
        parent_fqn: str,
        symbols: list[Symbol],
        imports: list[Import],
        call_sites: list[CallSite],
    ) -> None:
        name_node = field_node.child_by_field_name("name")
        value_node = field_node.child_by_field_name("value")
        if name_node is None:
            return
        name = _text(name_node, source)
        if value_node is not None and value_node.type in {
            "arrow_function",
            "function_expression",
        }:
            fqn = qualify(parent_fqn, name)
            symbols.append(
                Symbol(
                    fqn=fqn,
                    name=name,
                    kind=SymbolKind.METHOD,
                    lang=self.lang,
                    file_path=file_path,
                    span=_span(field_node),
                    signature=_signature(field_node, source),
                    docstring=_preceding_jsdoc(field_node, source),
                    visibility=_visibility(name),
                    parent_fqn=parent_fqn,
                )
            )
            body = value_node.child_by_field_name("body")
            if body is not None:
                self._walk(
                    node=body,
                    source=source,
                    file_path=file_path,
                    parent_fqn=fqn,
                    parent_kind_is_class=False,
                    symbols=symbols,
                    imports=imports,
                    call_sites=call_sites,
                )

    # --------------------------------------------------------------- imports

    def _extract_imports(
        self, node: Node, source: bytes, file_path: str
    ) -> Iterator[Import]:
        # `source` field on import_statement holds the literal string node.
        source_node = node.child_by_field_name("source")
        module_fqn = _string_literal_value(source_node, source) if source_node else ""
        if module_fqn is None:
            return
        raw = _text(node, source).strip()
        span = _span(node)

        # Find the import_clause (optional for side-effect imports).
        clause: Node | None = None
        for c in node.named_children:
            if c.type == "import_clause":
                clause = c
                break
        if clause is None:
            yield Import(
                file_path=file_path,
                raw=raw,
                local_name="",
                target_fqn=module_fqn,
                span=span,
            )
            return

        for c in clause.named_children:
            if c.type == "identifier":
                # default import: `import Foo from 'mod'`
                yield Import(
                    file_path=file_path,
                    raw=raw,
                    local_name=_text(c, source),
                    target_fqn=f"{module_fqn}.default",
                    span=span,
                )
            elif c.type == "namespace_import":
                # `import * as Foo from 'mod'`
                ident = next(
                    (g for g in c.named_children if g.type == "identifier"), None
                )
                if ident is not None:
                    yield Import(
                        file_path=file_path,
                        raw=raw,
                        local_name=_text(ident, source),
                        target_fqn=f"{module_fqn}.*",
                        span=span,
                    )
            elif c.type == "named_imports":
                for spec in c.named_children:
                    if spec.type != "import_specifier":
                        continue
                    name_node = spec.child_by_field_name("name")
                    alias_node = spec.child_by_field_name("alias")
                    if name_node is None:
                        continue
                    imported_name = _text(name_node, source)
                    local = (
                        _text(alias_node, source) if alias_node is not None else imported_name
                    )
                    yield Import(
                        file_path=file_path,
                        raw=raw,
                        local_name=local,
                        target_fqn=f"{module_fqn}.{imported_name}",
                        span=span,
                    )

    # --------------------------------------------------------------- calls

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
