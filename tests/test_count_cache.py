"""Unit tests for :mod:`code_spider.graph.count_cache`.

Focus on cache semantics; no Neo4j is touched. The schema tool's end-to-end
use of the cache is covered separately in ``test_get_graph_schema.py``.
"""

from __future__ import annotations

import time

from code_spider.graph.count_cache import (
    CountCache,
    CountEntry,
    get_cache,
    invalidate_workspace,
    set_cache,
)


def _entry(value: int = 1, scope: str = "workspace") -> CountEntry:
    return CountEntry(value=value, scope=scope)  # type: ignore[arg-type]


class TestGetOrCompute:
    def test_miss_then_hit_returns_same_cached_value(self) -> None:
        cache = CountCache(ttl_s=30.0)
        calls = {"n": 0}

        def fetch() -> CountEntry:
            calls["n"] += 1
            return _entry(value=42)

        first = cache.get_or_compute(
            workspace_id="demo",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=fetch,
        )
        second = cache.get_or_compute(
            workspace_id="demo",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=fetch,
        )
        assert first.value == 42
        assert second.value == 42
        # ``fetch`` ran exactly once.
        assert calls["n"] == 1

    def test_force_refresh_skips_cache(self) -> None:
        cache = CountCache(ttl_s=30.0)
        calls = {"n": 0}

        def fetch() -> CountEntry:
            calls["n"] += 1
            return _entry(value=calls["n"])

        first = cache.get_or_compute(
            workspace_id=None,
            kind="rel",
            name="CALLS",
            scoped=False,
            fetch=fetch,
        )
        forced = cache.get_or_compute(
            workspace_id=None,
            kind="rel",
            name="CALLS",
            scoped=False,
            fetch=fetch,
            force_refresh=True,
        )
        assert first.value == 1
        assert forced.value == 2
        assert calls["n"] == 2

    def test_ttl_zero_disables_caching(self) -> None:
        cache = CountCache(ttl_s=0.0)
        calls = {"n": 0}

        def fetch() -> CountEntry:
            calls["n"] += 1
            return _entry(value=calls["n"])

        for _ in range(3):
            cache.get_or_compute(
                workspace_id="demo",
                kind="node",
                name="File",
                scoped=True,
                fetch=fetch,
            )
        assert calls["n"] == 3
        assert len(cache) == 0

    def test_scope_is_part_of_key(self) -> None:
        cache = CountCache(ttl_s=30.0)
        cache.get_or_compute(
            workspace_id="demo",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=lambda: _entry(1, "workspace"),
        )
        # Same label, different scope → separate cache key, fetch runs again.
        result = cache.get_or_compute(
            workspace_id="demo",
            kind="node",
            name="Symbol",
            scoped=False,
            fetch=lambda: _entry(99, "global"),
        )
        assert result.value == 99
        assert result.scope == "global"
        assert len(cache) == 2


class TestExpiry:
    def test_entry_expires_after_ttl(self) -> None:
        cache = CountCache(ttl_s=0.05)  # 50 ms
        calls = {"n": 0}

        def fetch() -> CountEntry:
            calls["n"] += 1
            return _entry(value=calls["n"])

        cache.get_or_compute(
            workspace_id="demo",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=fetch,
        )
        time.sleep(0.07)
        cache.get_or_compute(
            workspace_id="demo",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=fetch,
        )
        assert calls["n"] == 2


class TestInvalidate:
    def test_invalidate_drops_workspace_and_global_entries(self) -> None:
        cache = CountCache(ttl_s=300.0)
        cache.get_or_compute(
            workspace_id="demo",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=lambda: _entry(1),
        )
        cache.get_or_compute(
            workspace_id="other",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=lambda: _entry(2),
        )
        cache.get_or_compute(
            workspace_id=None,
            kind="rel",
            name="CALLS",
            scoped=False,
            fetch=lambda: _entry(3, "global"),
        )

        dropped = cache.invalidate("demo", trigger="full")
        # ``demo`` + the global entry get dropped; ``other`` stays.
        assert dropped == 2
        assert len(cache) == 1

    def test_invalidate_no_op_when_nothing_matches(self) -> None:
        cache = CountCache(ttl_s=300.0)
        cache.get_or_compute(
            workspace_id="other",
            kind="node",
            name="Symbol",
            scoped=True,
            fetch=lambda: _entry(1),
        )
        # Note: ``None`` always matches because global entries are dropped
        # on any workspace invalidation. We avoid that here by using a
        # workspace name that doesn't exist.
        dropped = cache.invalidate("demo", trigger="manual")
        # The 'other' entry survives; the global None bucket is empty.
        assert dropped == 0
        assert len(cache) == 1


class TestSingleton:
    def test_set_cache_replaces_singleton(self) -> None:
        original = get_cache()
        try:
            replacement = CountCache(ttl_s=1.0)
            set_cache(replacement)
            assert get_cache() is replacement
            assert get_cache().ttl_s == 1.0
        finally:
            set_cache(original)

    def test_invalidate_workspace_convenience_targets_singleton(self) -> None:
        original = get_cache()
        try:
            cache = CountCache(ttl_s=300.0)
            set_cache(cache)
            cache.get_or_compute(
                workspace_id="demo",
                kind="node",
                name="File",
                scoped=True,
                fetch=lambda: _entry(1),
            )
            dropped = invalidate_workspace("demo", trigger="full")
            assert dropped == 1
            assert len(cache) == 0
        finally:
            set_cache(original)
