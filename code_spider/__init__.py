"""Code Spider — centralized codebase knowledge graph for AI coding agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("code-spider")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
