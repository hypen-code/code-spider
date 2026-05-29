"""Manifest schema and loader tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from code_spider.workspace.manifest import load_manifest, manifest_sha


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "workspaces.yaml"
    p.write_text(dedent(content), encoding="utf-8")
    return p


def test_load_minimal_local_manifest(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        version: 1
        workspaces:
          - id: demo
            name: Demo
            repos:
              - name: app
                path: .
                languages: [python]
        """,
    )
    m = load_manifest(p)
    assert len(m.workspaces) == 1
    w = m.workspace("demo")
    assert w.repos[0].path == "."
    assert w.repos[0].languages == ["python"]


def test_manifest_requires_url_xor_path(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        version: 1
        workspaces:
          - id: demo
            name: Demo
            repos:
              - name: app
                url: git@github.com:acme/app.git
                path: .
                languages: [python]
        """,
    )
    with pytest.raises(ValidationError, match="exactly one of"):
        load_manifest(p)


def test_manifest_rejects_duplicate_workspace_id(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        version: 1
        workspaces:
          - id: dup
            name: a
            repos: [{name: r, path: ., languages: [python]}]
          - id: dup
            name: b
            repos: [{name: r, path: ., languages: [python]}]
        """,
    )
    with pytest.raises(ValidationError, match="duplicate workspace id"):
        load_manifest(p)


def test_manifest_sha_is_stable(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        version: 1
        workspaces:
          - id: demo
            name: Demo
            repos: [{name: app, path: ., languages: [python]}]
        """,
    )
    m1 = load_manifest(p)
    m2 = load_manifest(p)
    assert manifest_sha(m1) == manifest_sha(m2)
    assert len(manifest_sha(m1)) == 64  # sha256 hex


def test_workspace_lookup_raises_on_missing(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        version: 1
        workspaces:
          - id: only
            name: Only
            repos: [{name: r, path: ., languages: [python]}]
        """,
    )
    m = load_manifest(p)
    with pytest.raises(KeyError):
        m.workspace("missing")
