"""The six resolution strategies, ordered by confidence (high to low).

Each strategy is a pure function with the same signature::

    def strategy(
        *,
        call_text: str,
        caller_fqn: str,
        caller_repo: str,
        file_imports: dict[str, str],
        index: SymbolIndex,
    ) -> ResolutionAttempt | None

It returns a :class:`ResolutionAttempt` if it can map the call to a single
Symbol, otherwise ``None`` so the cascade can fall through to the next one.
"""

from __future__ import annotations

from dataclasses import dataclass

from code_spider.resolver.index import SymbolIndex, SymbolRef


@dataclass(frozen=True, slots=True)
class ResolutionAttempt:
    """One strategy's verdict about a call site."""

    callee_repo: str
    callee_fqn: str
    confidence: float
    strategy: str


# Per-strategy confidences. Externalised so callers can adjust agent thresholds.
CONF_SAME_MODULE = 0.95
CONF_IMPORT_SUFFIX = 0.90
CONF_UNIQUE_GLOBAL = 0.85
CONF_SUFFIX_DISTANCE = 0.75
CONF_SYNTACTIC_FALLBACK = 0.55
# Unresolved calls do not emit an edge; the constant exists for parity with
# the design plan and for agents that want to surface attempts.
CONF_UNRESOLVED = 0.30


def _module_of(fqn: str) -> str:
    """The enclosing module for a symbol FQN — drop the last segment."""
    if "." not in fqn:
        return ""
    return fqn.rsplit(".", 1)[0]


def _split_call(call_text: str) -> tuple[list[str], str]:
    """Return ``(receiver_segments, simple_name)``.

    For ``foo()``                     -> (``[]``, ``"foo"``)
    For ``mod.foo()``                 -> (``["mod"]``, ``"foo"``)
    For ``a.b.c()``                   -> (``["a", "b"]``, ``"c"``)
    For ``self.client.get()``         -> (``["self", "client"]``, ``"get"``)
    """
    parts = call_text.split(".")
    return parts[:-1], parts[-1]


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


# -------------------------------------------------------------- strategies


def same_module(
    *,
    call_text: str,
    caller_fqn: str,
    caller_repo: str,
    file_imports: dict[str, str],
    index: SymbolIndex,
) -> ResolutionAttempt | None:
    receivers, name = _split_call(call_text)
    if receivers:
        return None
    module = _module_of(caller_fqn)
    if not module:
        return None
    fqn = f"{module}.{name}"
    sym = index.lookup_fqn(caller_repo, fqn)
    if sym is None:
        return None
    return ResolutionAttempt(
        callee_repo=caller_repo,
        callee_fqn=sym.fqn,
        confidence=CONF_SAME_MODULE,
        strategy="same-module",
    )


def import_suffix(
    *,
    call_text: str,
    caller_fqn: str,
    caller_repo: str,
    file_imports: dict[str, str],
    index: SymbolIndex,
) -> ResolutionAttempt | None:
    receivers, name = _split_call(call_text)
    head = receivers[0] if receivers else name

    if head not in file_imports:
        return None
    target_root = file_imports[head]

    # ``receivers[0]`` was the local import name; we strip it and use the
    # imported target as the prefix. Bare ``foo()`` -> the import target itself.
    candidate = (
        ".".join([target_root, *receivers[1:], name]) if receivers else target_root
    )

    direct = index.lookup_fqn(caller_repo, candidate)
    if direct is None:
        ref = index.find_fqn(candidate)
        if ref is None:
            return None
        repo, fqn = ref.repo, ref.fqn
    else:
        repo, fqn = caller_repo, direct.fqn

    return ResolutionAttempt(
        callee_repo=repo,
        callee_fqn=fqn,
        confidence=CONF_IMPORT_SUFFIX,
        strategy="import-suffix",
    )


def unique_global(
    *,
    call_text: str,
    caller_fqn: str,
    caller_repo: str,
    file_imports: dict[str, str],
    index: SymbolIndex,
) -> ResolutionAttempt | None:
    _, name = _split_call(call_text)
    candidates = index.lookup_simple_name(name)
    if len(candidates) != 1:
        return None
    ref = candidates[0]
    return ResolutionAttempt(
        callee_repo=ref.repo,
        callee_fqn=ref.fqn,
        confidence=CONF_UNIQUE_GLOBAL,
        strategy="unique-global",
    )


def suffix_by_distance(
    *,
    call_text: str,
    caller_fqn: str,
    caller_repo: str,
    file_imports: dict[str, str],
    index: SymbolIndex,
) -> ResolutionAttempt | None:
    _, name = _split_call(call_text)
    candidates = index.lookup_simple_name(name)
    if len(candidates) <= 1:
        return None
    caller_parts = caller_fqn.split(".")
    # Score by length of common FQN prefix; tie-break by total FQN length.
    def score(ref: SymbolRef) -> tuple[int, int]:
        parts = ref.fqn.split(".")
        return (_common_prefix(parts, caller_parts), -len(parts))

    best = max(candidates, key=score)
    cp, _ = score(best)
    if cp == 0:
        return None
    return ResolutionAttempt(
        callee_repo=best.repo,
        callee_fqn=best.fqn,
        confidence=CONF_SUFFIX_DISTANCE,
        strategy="suffix-distance",
    )


def syntactic_fallback(
    *,
    call_text: str,
    caller_fqn: str,
    caller_repo: str,
    file_imports: dict[str, str],
    index: SymbolIndex,
) -> ResolutionAttempt | None:
    """Last-ditch: pick the unique FQN whose suffix matches the call's full chain."""
    receivers, name = _split_call(call_text)
    if not receivers:
        return None
    suffix = ".".join([*receivers, name])
    matches: list[SymbolRef] = []
    for ref in index.lookup_simple_name(name):
        if ref.fqn.endswith("." + suffix) or ref.fqn == suffix:
            matches.append(ref)
            if len(matches) > 1:
                return None  # ambiguous
    if len(matches) != 1:
        return None
    only = matches[0]
    return ResolutionAttempt(
        callee_repo=only.repo,
        callee_fqn=only.fqn,
        confidence=CONF_SYNTACTIC_FALLBACK,
        strategy="syntactic-fallback",
    )


# Strategy chain, highest confidence first. The cascade applies them in order
# and stops at the first hit.
STRATEGY_CHAIN = (
    same_module,
    import_suffix,
    unique_global,
    suffix_by_distance,
    syntactic_fallback,
)
