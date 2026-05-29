"""Workspace-wide symbol and import indexes used by the resolver cascade.

These are pure in-memory data structures populated once per indexer run,
after every adapter has produced its :class:`FileRecord`s. They are read by
:mod:`code_spider.resolver.strategies` and :mod:`code_spider.resolver.cascade`
to map raw call-site text into concrete callee FQNs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from code_spider.symbols.model import Import, Symbol, WorkspaceParseBundle


@dataclass(frozen=True, slots=True)
class SymbolRef:
    """A pointer to a Symbol in the workspace index."""

    repo: str
    fqn: str


class SymbolIndex:
    """Multi-key lookup table for every Symbol parsed in a workspace."""

    __slots__ = ("_by_fqn", "_by_name", "_fqns_by_repo")

    def __init__(self) -> None:
        self._by_fqn: dict[tuple[str, str], Symbol] = {}
        self._by_name: dict[str, list[SymbolRef]] = defaultdict(list)
        self._fqns_by_repo: dict[str, set[str]] = defaultdict(set)

    def add(self, repo: str, sym: Symbol) -> None:
        key = (repo, sym.fqn)
        if key in self._by_fqn:
            return
        self._by_fqn[key] = sym
        self._by_name[sym.name].append(SymbolRef(repo=repo, fqn=sym.fqn))
        self._fqns_by_repo[repo].add(sym.fqn)

    def lookup_fqn(self, repo: str, fqn: str) -> Symbol | None:
        """Exact lookup in a single repo first, then any repo (workspace fallback)."""
        direct = self._by_fqn.get((repo, fqn))
        if direct is not None:
            return direct
        for (r, f), s in self._by_fqn.items():
            if f == fqn and r != repo:
                return s
        return None

    def find_fqn(self, fqn: str) -> SymbolRef | None:
        """Locate any repo that defines exactly ``fqn``."""
        for (r, f), _ in self._by_fqn.items():
            if f == fqn:
                return SymbolRef(repo=r, fqn=f)
        return None

    def lookup_simple_name(self, name: str) -> list[SymbolRef]:
        return list(self._by_name.get(name, ()))

    def get(self, ref: SymbolRef) -> Symbol | None:
        return self._by_fqn.get((ref.repo, ref.fqn))

    def __len__(self) -> int:
        return len(self._by_fqn)


class ImportMap:
    """Per-(repo, file) ``local_name -> imported_target_fqn`` map."""

    __slots__ = ("_maps",)

    def __init__(self) -> None:
        self._maps: dict[tuple[str, str], dict[str, str]] = {}

    def add_file(self, repo: str, file_path: str, imports: list[Import]) -> None:
        m = self._maps.setdefault((repo, file_path), {})
        for imp in imports:
            if imp.local_name and imp.local_name != "*":
                # Last-write-wins is fine; aliased imports take precedence.
                m[imp.local_name] = imp.target_fqn

    def for_file(self, repo: str, file_path: str) -> dict[str, str]:
        return self._maps.get((repo, file_path), {})


def build_indexes(bundle: WorkspaceParseBundle) -> tuple[SymbolIndex, ImportMap]:
    """Materialise the workspace-wide :class:`SymbolIndex` + :class:`ImportMap`."""
    idx = SymbolIndex()
    imap = ImportMap()
    for pr in bundle.repos:
        for f in pr.files:
            for sym in f.symbols:
                idx.add(pr.repo_name, sym)
            imap.add_file(pr.repo_name, f.repo_relative_path, f.imports)
    return idx, imap
