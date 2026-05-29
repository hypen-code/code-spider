"""Resolver cascade tests — strategies + workspace-wide resolution."""

from __future__ import annotations

from textwrap import dedent

from code_spider.parser import get_adapter
from code_spider.resolver import build_indexes, resolve_workspace
from code_spider.resolver.strategies import (
    CONF_IMPORT_SUFFIX,
    CONF_SAME_MODULE,
    CONF_UNIQUE_GLOBAL,
    import_suffix,
    same_module,
    suffix_by_distance,
    syntactic_fallback,
    unique_global,
)
from code_spider.symbols.model import ParseResult, WorkspaceParseBundle


def _parse(source: str, path: str = "pkg/module.py"):
    return get_adapter("python").parse_file(path, dedent(source).encode("utf-8"))


def _bundle_from(source: str, path: str = "pkg/module.py") -> WorkspaceParseBundle:
    fr = _parse(source, path)
    pr = ParseResult(
        workspace_id="ws", repo_name="repo", commit_sha="sha", files=[fr]
    )
    bundle = WorkspaceParseBundle(
        workspace_id="ws", workspace_name="Ws", manifest_sha="x"
    )
    bundle.repos.append(pr)
    return bundle


def test_same_module_strategy_resolves_sibling_function() -> None:
    bundle = _bundle_from(
        """
        def helper():
            return 1

        def caller():
            return helper()
        """
    )
    stats = resolve_workspace(bundle)
    pr = bundle.repos[0]
    same_module_calls = [rc for rc in pr.resolved_calls if rc.strategy == "same-module"]
    assert any(rc.callee_fqn == "pkg.module.helper" for rc in same_module_calls)
    assert stats.get("same-module", 0) >= 1


def test_import_suffix_strategy_resolves_through_local_alias() -> None:
    # Use ``pkg/`` (not ``lib/`` or ``src/``) so file_to_module_fqn does
    # not strip the prefix and the import target literally matches.
    fr_target = _parse(
        """
        def doit():
            return 1
        """,
        "pkg/things.py",
    )
    fr_caller = _parse(
        """
        from pkg.things import doit as d

        def run():
            return d()
        """,
        "app/runner.py",
    )
    pr = ParseResult(
        workspace_id="ws", repo_name="repo", commit_sha="sha", files=[fr_target, fr_caller]
    )
    bundle = WorkspaceParseBundle(
        workspace_id="ws", workspace_name="Ws", manifest_sha="x"
    )
    bundle.repos.append(pr)
    resolve_workspace(bundle)

    matched = [
        rc
        for rc in pr.resolved_calls
        if rc.callee_fqn == "pkg.things.doit"
    ]
    assert matched, (
        "expected resolved CALL into pkg.things.doit, "
        f"got: {pr.resolved_calls}"
    )
    assert matched[0].strategy == "import-suffix"
    assert matched[0].confidence == CONF_IMPORT_SUFFIX


def test_unique_global_strategy_resolves_when_only_one_candidate() -> None:
    bundle = _bundle_from(
        """
        def caller():
            return unique_name()
        """
    )
    # Hand-craft a second file with a single matching symbol.
    other = _parse(
        """
        def unique_name():
            return 0
        """,
        "other/util.py",
    )
    bundle.repos[0].files.append(other)
    resolve_workspace(bundle)

    pr = bundle.repos[0]
    hits = [rc for rc in pr.resolved_calls if rc.callee_fqn == "other.util.unique_name"]
    assert hits
    assert hits[0].strategy in {"unique-global", "import-suffix"}


def test_strategy_unit_invariants() -> None:
    """The strategy chain returns ``None`` or a ResolutionAttempt; no exceptions."""
    bundle = _bundle_from(
        """
        def foo():
            return bar()
        """
    )
    index, imap = build_indexes(bundle)
    # ``bar`` is not present anywhere — every strategy must return None.
    strategies = (
        same_module,
        import_suffix,
        unique_global,
        suffix_by_distance,
        syntactic_fallback,
    )
    for strat in strategies:
        result = strat(
            call_text="bar",
            caller_fqn="pkg.module.foo",
            caller_repo="repo",
            file_imports=imap.for_file("repo", "pkg/module.py"),
            index=index,
        )
        assert result is None


def test_confidence_priorities_are_descending() -> None:
    assert CONF_SAME_MODULE > CONF_IMPORT_SUFFIX > CONF_UNIQUE_GLOBAL
