"""TypeScript / JavaScript adapter tests."""

from __future__ import annotations

from textwrap import dedent

from code_spider.parser import get_adapter
from code_spider.symbols.model import SymbolKind


def _parse_ts(source: str, path: str = "src/sample.ts"):
    return get_adapter("typescript").parse_file(path, dedent(source).encode("utf-8"))


def _parse_js(source: str, path: str = "sample.js"):
    return get_adapter("javascript").parse_file(path, dedent(source).encode("utf-8"))


def test_ts_extracts_top_level_function_and_class() -> None:
    fr = _parse_ts(
        """
        /** Greet someone. */
        export function greet(name: string): string {
          return `hi ${name}`;
        }

        export class Greeter {
          prefix: string;
          constructor(prefix: string) { this.prefix = prefix; }
          greet(name: string): string { return `${this.prefix} ${name}`; }
        }
        """
    )
    kinds = {(s.kind, s.name) for s in fr.symbols}
    assert (SymbolKind.FUNCTION, "greet") in kinds
    assert (SymbolKind.CLASS, "Greeter") in kinds
    assert (SymbolKind.METHOD, "greet") in kinds


def test_ts_captures_interfaces_and_type_aliases() -> None:
    fr = _parse_ts(
        """
        export interface User { id: string; name: string; }
        export type ID = string | number;
        """
    )
    by_kind = {s.kind: s.name for s in fr.symbols}
    assert by_kind.get(SymbolKind.INTERFACE) == "User"
    assert by_kind.get(SymbolKind.TYPE_ALIAS) == "ID"


def test_ts_extracts_arrow_function_via_lexical_decl() -> None:
    fr = _parse_ts(
        """
        const helper = async (x: number): Promise<number> => x * 2;
        const constant = 42;
        """
    )
    funcs = {s.name: s.kind for s in fr.symbols}
    assert funcs.get("helper") == SymbolKind.FUNCTION
    assert funcs.get("constant") == SymbolKind.VARIABLE


def test_ts_captures_named_default_and_namespace_imports() -> None:
    fr = _parse_ts(
        """
        import axios from 'axios';
        import { Controller, Get } from '@nestjs/common';
        import * as fs from 'fs';
        """
    )
    pairs = {(i.local_name, i.target_fqn) for i in fr.imports}
    assert ("axios", "axios.default") in pairs
    assert ("Controller", "@nestjs/common.Controller") in pairs
    assert ("Get", "@nestjs/common.Get") in pairs
    assert ("fs", "fs.*") in pairs


def test_ts_records_call_sites_with_caller_fqn() -> None:
    fr = _parse_ts(
        """
        function outer() {
          inner();
          obj.method();
        }
        outer();
        """
    )
    callers = {c.caller_fqn for c in fr.call_sites}
    assert "src.sample.outer" in callers
    assert "src.sample" in callers  # module-scope call


def test_js_handles_class_with_methods() -> None:
    fr = _parse_js(
        """
        class Adder {
          add(a, b) { return a + b; }
        }
        export default Adder;
        """,
        path="adder.js",
    )
    names = {s.name for s in fr.symbols}
    assert "Adder" in names
    assert "add" in names


def test_nextjs_app_router_routes_inferred_from_path() -> None:
    fr = _parse_ts(
        """
        export async function GET(req: Request) { return new Response(); }
        export async function POST(req: Request) { return new Response(); }
        """,
        path="app/api/users/[id]/route.ts",
    )
    routes = {(r.method, r.path, r.framework) for r in fr.routes}
    assert ("GET", "/api/users/{}", "nextjs-app") in routes
    assert ("POST", "/api/users/{}", "nextjs-app") in routes


def test_ts_axios_and_fetch_clients_captured() -> None:
    fr = _parse_ts(
        """
        import axios from 'axios';
        async function load(id: string) {
          await axios.get(`/users/${id}`);
          await fetch('/users', { method: 'POST' });
        }
        """
    )
    method_path = {(h.method, h.path_template) for h in fr.http_clients}
    assert ("GET", "/users/{}") in method_path
    assert ("POST", "/users") in method_path
