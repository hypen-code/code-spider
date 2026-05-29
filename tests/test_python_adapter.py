"""Tree-sitter Python adapter tests."""

from __future__ import annotations

from textwrap import dedent

from code_spider.parser.python_adapter import PythonAdapter
from code_spider.symbols.model import SymbolKind


def _parse(source: str, path: str = "pkg/module.py"):
    return PythonAdapter().parse_file(path, dedent(source).encode("utf-8"))


def test_extracts_top_level_function() -> None:
    fr = _parse(
        """
        def hello(name: str) -> str:
            \"\"\"Greet someone.\"\"\"
            return f"hi {name}"
        """
    )
    funcs = [s for s in fr.symbols if s.kind == SymbolKind.FUNCTION]
    assert len(funcs) == 1
    f = funcs[0]
    assert f.name == "hello"
    assert f.fqn == "pkg.module.hello"
    assert "def hello" in f.signature
    assert f.docstring == "Greet someone."
    assert f.span.start_line >= 1
    assert f.parent_fqn == "pkg.module"
    assert f.visibility == "public"


def test_extracts_class_and_methods_with_correct_kinds() -> None:
    fr = _parse(
        """
        class Greeter:
            \"\"\"A greeter.\"\"\"
            def __init__(self, prefix: str):
                self.prefix = prefix

            def greet(self, name: str) -> str:
                return f"{self.prefix} {name}"
        """
    )
    by_kind: dict[str, list[str]] = {}
    for s in fr.symbols:
        by_kind.setdefault(str(s.kind), []).append(s.fqn)

    assert "class" in by_kind and "pkg.module.Greeter" in by_kind["class"]
    methods = sorted(by_kind.get("method", []))
    assert methods == [
        "pkg.module.Greeter.__init__",
        "pkg.module.Greeter.greet",
    ]


def test_extracts_decorated_definition() -> None:
    fr = _parse(
        """
        from functools import lru_cache

        @lru_cache(maxsize=8)
        def memoised(x: int) -> int:
            return x * 2
        """
    )
    names = {s.name for s in fr.symbols}
    assert "memoised" in names


def test_extracts_imports() -> None:
    fr = _parse(
        """
        import os
        import json as j
        from pathlib import Path
        from typing import Annotated as A, Literal
        """
    )
    targets = {(i.local_name, i.target_fqn) for i in fr.imports}
    assert ("os", "os") in targets
    assert ("j", "json") in targets
    assert ("Path", "pathlib.Path") in targets
    assert ("A", "typing.Annotated") in targets
    assert ("Literal", "typing.Literal") in targets


def test_records_call_sites_with_caller_fqn() -> None:
    fr = _parse(
        """
        def outer():
            inner_call()
            obj.method()

        outer()
        """
    )
    callers = {c.caller_fqn for c in fr.call_sites}
    # The bare ``outer()`` call at module scope has the module as its caller.
    assert "pkg.module.outer" in callers
    assert "pkg.module" in callers


def test_private_visibility_marker() -> None:
    fr = _parse(
        """
        def _internal():
            pass

        def __dunder__():
            pass

        def __really_private():
            pass
        """
    )
    vis = {s.name: s.visibility for s in fr.symbols}
    assert vis["_internal"] == "protected"
    assert vis["__dunder__"] == "public"
    assert vis["__really_private"] == "private"


def test_handles_syntax_error_gracefully() -> None:
    fr = _parse(
        """
        def good():
            return 1

        def bad(:
            return 2
        """
    )
    # Tree-sitter is error-tolerant; we should still see the good function.
    names = {s.name for s in fr.symbols}
    assert "good" in names
