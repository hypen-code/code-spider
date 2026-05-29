"""Language adapter protocol + registry.

Each supported language (python, typescript, javascript) provides a class
implementing :class:`LanguageAdapter`. The registry below resolves language
strings (matching ``RepoConfig.languages`` entries) to adapter instances.

Adding a new language:
    1. Create ``code_spider/parser/<lang>_adapter.py`` defining
       ``class <Lang>Adapter(LanguageAdapter)``.
    2. Register it via :func:`register_adapter` at module import time, or add
       a lazy import in :func:`get_adapter`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from code_spider.symbols.model import FileRecord


@runtime_checkable
class LanguageAdapter(Protocol):
    """Parses one file's source into a :class:`FileRecord`."""

    lang: str
    extensions: tuple[str, ...]

    def parse_file(self, repo_relative_path: str, source: bytes) -> FileRecord:
        """Parse ``source`` (raw file bytes) and return a populated FileRecord."""
        ...


_REGISTRY: dict[str, LanguageAdapter] = {}


def register_adapter(adapter: LanguageAdapter) -> None:
    """Register an adapter under its ``lang`` key. Idempotent for the same instance."""
    _REGISTRY[adapter.lang] = adapter


def get_adapter(lang: str) -> LanguageAdapter:
    """Return the adapter for ``lang``. Lazy-loads built-in adapters on demand."""
    if lang not in _REGISTRY:
        _lazy_load(lang)
    if lang not in _REGISTRY:
        raise KeyError(f"no language adapter registered for '{lang}'")
    return _REGISTRY[lang]


def _lazy_load(lang: str) -> None:
    """Import a built-in adapter on first use to avoid loading every grammar eagerly."""
    if lang == "python":
        from code_spider.parser.python_adapter import PythonAdapter

        register_adapter(PythonAdapter())
    elif lang == "typescript":
        from code_spider.parser.typescript_adapter import TypeScriptAdapter

        register_adapter(TypeScriptAdapter())
    elif lang == "javascript":
        from code_spider.parser.javascript_adapter import JavaScriptAdapter

        register_adapter(JavaScriptAdapter())
