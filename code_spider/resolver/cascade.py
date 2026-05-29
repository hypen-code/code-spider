"""Workspace-wide call resolver: applies the strategy chain to every CallSite.

Entry point: :func:`resolve_workspace`. Mutates each :class:`ParseResult`'s
``resolved_calls`` list in place and additionally populates each
:class:`Import`'s ``resolved_fqn`` field whenever the import target is
present in the workspace symbol index.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from code_spider.logging_setup import get_logger
from code_spider.resolver.index import SymbolIndex, build_indexes
from code_spider.resolver.strategies import STRATEGY_CHAIN
from code_spider.symbols.model import CallSite, ResolvedCall, WorkspaceParseBundle

_log = get_logger(__name__)


def resolve_workspace(bundle: WorkspaceParseBundle) -> dict[str, int]:
    """Resolve every CallSite in ``bundle``; mutate it in place.

    Returns a stats dict ``{strategy_name -> count}`` for logging/reporting.
    """
    index, imap = build_indexes(bundle)
    _log.info(
        "resolver indexes built",
        symbols=len(index),
        files=sum(len(p.files) for p in bundle.repos),
    )

    counter: Counter[str] = Counter()

    for pr in bundle.repos:
        # Mutate the (mutable) list inside the frozen dataclass.
        for f in pr.files:
            for i, imp in enumerate(f.imports):
                ref = index.find_fqn(imp.target_fqn)
                if ref is not None:
                    f.imports[i] = replace(imp, resolved_fqn=ref.fqn)
                    counter["import-resolved"] += 1

            for cs in f.call_sites:
                resolved = _resolve_call_site(
                    cs=cs,
                    caller_repo=pr.repo_name,
                    index=index,
                    file_imports=imap.for_file(pr.repo_name, f.repo_relative_path),
                )
                if resolved is not None:
                    pr.resolved_calls.append(resolved)
                    counter[resolved.strategy] += 1
                else:
                    counter["unresolved"] += 1

    _log.info("resolution complete", **counter)
    return dict(counter)


def _resolve_call_site(
    *,
    cs: CallSite,
    caller_repo: str,
    index: SymbolIndex,
    file_imports: dict[str, str],
) -> ResolvedCall | None:
    for strategy in STRATEGY_CHAIN:
        attempt = strategy(
            call_text=cs.call_text,
            caller_fqn=cs.caller_fqn,
            caller_repo=caller_repo,
            file_imports=file_imports,
            index=index,
        )
        if attempt is not None:
            return ResolvedCall(
                caller_fqn=cs.caller_fqn,
                callee_fqn=attempt.callee_fqn,
                callee_repo=attempt.callee_repo,
                confidence=attempt.confidence,
                strategy=attempt.strategy,
                file_path=cs.file_path,
                span=cs.span,
            )
    return None
