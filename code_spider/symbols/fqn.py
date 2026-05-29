"""Fully-qualified name (FQN) helpers.

FQNs are namespaced by ``(workspace_id, repo_name)`` at the Neo4j constraint level,
so this module only handles the *intra-repo* dotted-path portion.

For Python:
    src/code_spider/parser/python_adapter.py  -> code_spider.parser.python_adapter
    code_spider/parser/python_adapter.py      -> code_spider.parser.python_adapter
    a/b/__init__.py                           -> a.b

For TS/JS the convention is the same with the file's *extension* stripped.
"""

from __future__ import annotations

from pathlib import PurePosixPath

_PY_SRC_PREFIXES = ("src/", "lib/")
_PY_EXT = {".py", ".pyi"}
_TS_EXT = {".ts", ".tsx", ".mts", ".cts"}
_JS_EXT = {".js", ".jsx", ".mjs", ".cjs"}


def file_to_module_fqn(repo_relative_path: str, lang: str) -> str:
    """Translate a repo-relative file path into a dotted module FQN.

    ``lang`` is one of ``"python"``, ``"typescript"``, ``"javascript"``.
    """
    p = PurePosixPath(repo_relative_path)
    parts = list(p.parts)
    if not parts:
        return ""

    if lang == "python":
        # Strip common Python source roots so ``src/code_spider/...`` and
        # ``code_spider/...`` produce the same FQN.
        while parts and f"{parts[0]}/" in _PY_SRC_PREFIXES:
            parts.pop(0)
        if not parts:
            return ""
        last = PurePosixPath(parts[-1])
        if last.suffix not in _PY_EXT:
            return ".".join(parts)
        stem = last.stem
        if stem == "__init__":
            parts = parts[:-1]
        else:
            parts[-1] = stem
        return ".".join(parts)

    last = PurePosixPath(parts[-1])

    if lang in {"typescript", "javascript"}:
        valid = _TS_EXT if lang == "typescript" else _JS_EXT
        if last.suffix not in valid:
            return "/".join(parts)
        stem = last.stem
        if stem == "index":
            parts = parts[:-1]
        else:
            parts[-1] = stem
        # JS/TS modules conventionally use slashes, but for graph FQNs we use dots
        # to keep a single namespace separator across languages.
        return ".".join(parts)

    return ".".join(parts)


def qualify(parent_fqn: str, name: str) -> str:
    """Join a parent FQN with a child name. Handles the empty-parent case."""
    return f"{parent_fqn}.{name}" if parent_fqn else name
