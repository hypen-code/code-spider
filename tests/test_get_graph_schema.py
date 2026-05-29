"""Unit tests for ``get_graph_schema``.

The orchestrator function is exercised end-to-end against the live Neo4j
in a separate smoke script. These tests cover the *pure* aggregators
that fold ``db.schema.*`` rows into the response shape, plus the
injection-proof label / rel-type guards on the count queries.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

# ``get_graph_schema`` is re-exported as a function on the parent package,
# so we pull the actual module via importlib.
gs_mod = importlib.import_module("code_spider.mcp.tools.get_graph_schema")
_aggregate_node_props = gs_mod._aggregate_node_props
_aggregate_rel_props = gs_mod._aggregate_rel_props
_aggregate_endpoints = gs_mod._aggregate_endpoints
_node_count_query = gs_mod._node_count_query
_rel_count_query = gs_mod._rel_count_query
_strip_quoted_type = gs_mod._strip_quoted_type


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


class TestAggregateNodeProps:
    def test_groups_properties_by_label_and_sorts(self) -> None:
        rows = [
            {
                "nodeType": ":`Symbol`",
                "nodeLabels": ["Symbol"],
                "propertyName": "fqn",
                "propertyTypes": ["String"],
                "mandatory": True,
            },
            {
                "nodeType": ":`Symbol`",
                "nodeLabels": ["Symbol"],
                "propertyName": "kind",
                "propertyTypes": ["String"],
                "mandatory": False,
            },
            {
                "nodeType": ":`File`",
                "nodeLabels": ["File"],
                "propertyName": "path",
                "propertyTypes": ["String"],
                "mandatory": True,
            },
        ]
        out = _aggregate_node_props(rows)
        assert set(out) == {"Symbol", "File"}
        assert [p["name"] for p in out["Symbol"]] == ["fqn", "kind"]
        assert out["Symbol"][0] == {
            "name": "fqn",
            "types": ["String"],
            "mandatory": True,
        }
        assert out["File"] == [
            {"name": "path", "types": ["String"], "mandatory": True}
        ]

    def test_label_with_no_properties_is_kept(self) -> None:
        rows = [
            {
                "nodeType": ":`Tagged`",
                "nodeLabels": ["Tagged"],
                "propertyName": None,
                "propertyTypes": [],
                "mandatory": False,
            }
        ]
        out = _aggregate_node_props(rows)
        assert out == {"Tagged": []}

    def test_multi_label_nodes_list_props_under_each_label(self) -> None:
        rows = [
            {
                "nodeLabels": ["Symbol", "Function"],
                "propertyName": "fqn",
                "propertyTypes": ["String"],
                "mandatory": True,
            }
        ]
        out = _aggregate_node_props(rows)
        assert out["Symbol"] == out["Function"]
        assert out["Symbol"][0]["name"] == "fqn"

    def test_dedupes_repeated_properties_for_multi_labelled_nodes(self) -> None:
        # In a real graph a single :Symbol:Function node will appear under
        # multiple nodeType rows. Each row repeats ``commit_sha`` under
        # ``Symbol`` and ``Function``. We must not return the duplicates.
        rows = [
            {
                "nodeLabels": ["Symbol", "Function"],
                "propertyName": "commit_sha",
                "propertyTypes": ["String"],
                "mandatory": False,
            },
            {
                "nodeLabels": ["Symbol", "Class"],
                "propertyName": "commit_sha",
                "propertyTypes": ["String"],
                "mandatory": True,
            },
            {
                "nodeLabels": ["Symbol"],
                "propertyName": "fqn",
                "propertyTypes": ["String"],
                "mandatory": True,
            },
        ]
        out = _aggregate_node_props(rows)
        # ``Symbol`` had ``commit_sha`` repeated; should appear only once.
        names = [p["name"] for p in out["Symbol"]]
        assert names == sorted(names)
        assert names.count("commit_sha") == 1
        # Mandatory only when mandatory in *every* combo — Symbol saw it
        # both mandatory=True and mandatory=False, so the merged answer is
        # ``False``.
        commit_sha = next(p for p in out["Symbol"] if p["name"] == "commit_sha")
        assert commit_sha["mandatory"] is False
        assert commit_sha["types"] == ["String"]


class TestAggregateRelProps:
    def test_strips_neo4j_quoting_on_rel_type(self) -> None:
        rows = [
            {
                "relType": ":`CALLS`",
                "propertyName": "confidence",
                "propertyTypes": ["Float"],
                "mandatory": False,
            },
            {
                "relType": ":`CALLS`",
                "propertyName": "kind",
                "propertyTypes": ["String"],
                "mandatory": False,
            },
        ]
        out = _aggregate_rel_props(rows)
        assert set(out) == {"CALLS"}
        assert [p["name"] for p in out["CALLS"]] == ["confidence", "kind"]

    def test_skips_rows_without_relType(self) -> None:
        rows = [
            {"relType": None, "propertyName": "x", "propertyTypes": ["Int"]}
        ]
        assert _aggregate_rel_props(rows) == {}


@pytest.mark.parametrize(
    "raw,expected",
    [
        (":`CALLS`", "CALLS"),
        (":CALLS", "CALLS"),
        ("`CALLS`", "CALLS"),
        ("CALLS", "CALLS"),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_strip_quoted_type(raw: Any, expected: str | None) -> None:
    assert _strip_quoted_type(raw) == expected


class _VirtualNode:
    """Quacks like a Neo4j virtual node returned by db.schema.visualization."""

    def __init__(self, element_id: str, name: str) -> None:
        self.element_id = element_id
        self._props = {"name": name}

    def __getitem__(self, k: str) -> Any:
        return self._props[k]

    def keys(self):
        return self._props.keys()


class _VirtualRel:
    def __init__(self, rt: str, start: _VirtualNode, end: _VirtualNode) -> None:
        self.type = rt
        self.start_node = start
        self.end_node = end


class TestAggregateEndpoints:
    def test_returns_deduplicated_start_end_pairs_per_rel_type(self) -> None:
        sym1 = _VirtualNode("n:1", "Symbol")
        sym2 = _VirtualNode("n:2", "Symbol")
        file_ = _VirtualNode("n:3", "File")
        rels = [
            _VirtualRel("CALLS", sym1, sym2),
            _VirtualRel("CALLS", sym1, sym2),  # duplicate — must dedupe
            _VirtualRel("DEFINES", file_, sym1),
        ]
        out = _aggregate_endpoints([sym1, sym2, file_], rels)
        assert out == {
            "CALLS": [{"start": "Symbol", "end": "Symbol"}],
            "DEFINES": [{"start": "File", "end": "Symbol"}],
        }

    def test_skips_rels_with_unknown_endpoints(self) -> None:
        # An endpoint we never registered under ``nodes``.
        sym = _VirtualNode("n:1", "Symbol")
        orphan = _VirtualNode("n:99", "Orphan")
        rels = [_VirtualRel("CALLS", sym, orphan)]
        out = _aggregate_endpoints([sym], rels)  # ``orphan`` not in nodes list
        assert out == {}


# --------------------------------------------------------------------------- #
# Injection-proofing for the dynamic count queries                            #
# --------------------------------------------------------------------------- #


class TestCountQuerySafety:
    @pytest.mark.parametrize(
        "label",
        ["Symbol", "File", "_Internal", "FooBar123"],
    )
    def test_node_count_accepts_safe_labels(self, label: str) -> None:
        q = _node_count_query(label, scoped=True)
        assert f"(n:`{label}`)" in q
        assert "$workspace_id" in q

    def test_node_count_global_skips_workspace_filter(self) -> None:
        q = _node_count_query("Workspace", scoped=False)
        assert "(n:`Workspace`)" in q
        assert "$workspace_id" not in q
        assert "WHERE" not in q

    @pytest.mark.parametrize(
        "bad",
        [
            "Symbol`) DETACH DELETE (n",   # backtick break-out
            "Symbol DROP TABLE",
            "Sym bol",                      # space
            "1Symbol",                      # leading digit
            "",
            "Symbol;CREATE",
        ],
    )
    def test_node_count_rejects_unsafe_labels(self, bad: str) -> None:
        with pytest.raises(ValueError, match="unsafe label"):
            _node_count_query(bad, scoped=True)

    def test_rel_count_query_scopes_on_start_node_workspace(self) -> None:
        q = _rel_count_query("CALLS", scoped=True)
        # Edges in this graph do not all carry ``workspace_id``; scope on
        # the start node instead, which we test here so a future refactor
        # cannot silently change the semantics.
        assert "[r:`CALLS`]" in q
        assert "a.workspace_id" in q

    def test_rel_count_global_skips_workspace_filter(self) -> None:
        q = _rel_count_query("CALLS", scoped=False)
        assert "[r:`CALLS`]" in q
        assert "$workspace_id" not in q

    @pytest.mark.parametrize(
        "bad",
        ["CALLS`) MATCH", "1REL", "", "REL TYPE"],
    )
    def test_rel_count_rejects_unsafe_types(self, bad: str) -> None:
        with pytest.raises(ValueError, match="unsafe relationship type"):
            _rel_count_query(bad, scoped=True)


# --------------------------------------------------------------------------- #
# Public tool signature                                                       #
# --------------------------------------------------------------------------- #


class TestPublicSignature:
    def test_accepts_force_refresh_kwarg(self) -> None:
        """``force_refresh`` is the documented cache-bypass knob."""
        sig = inspect.signature(gs_mod.get_graph_schema)
        params = sig.parameters
        assert "force_refresh" in params
        assert params["force_refresh"].default is False
        # ``workspace_id`` and ``include_counts`` stay backward-compatible.
        assert params["workspace_id"].default is None
        assert params["include_counts"].default is True
