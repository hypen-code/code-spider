"""FQN computation tests."""

from __future__ import annotations

import pytest

from code_spider.symbols.fqn import file_to_module_fqn, qualify


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("code_spider/parser/python_adapter.py", "code_spider.parser.python_adapter"),
        ("src/code_spider/cli.py", "code_spider.cli"),
        ("lib/util.py", "util"),
        ("code_spider/__init__.py", "code_spider"),
        ("a/b/__init__.py", "a.b"),
        ("standalone.py", "standalone"),
    ],
)
def test_python_module_fqn(path: str, expected: str) -> None:
    assert file_to_module_fqn(path, "python") == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/app.ts", "src.app"),
        ("src/index.ts", "src"),
        ("packages/api/src/server.tsx", "packages.api.src.server"),
    ],
)
def test_typescript_module_fqn(path: str, expected: str) -> None:
    assert file_to_module_fqn(path, "typescript") == expected


def test_qualify_handles_empty_parent() -> None:
    assert qualify("", "foo") == "foo"
    assert qualify("a.b", "c") == "a.b.c"
