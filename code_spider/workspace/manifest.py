"""Workspace manifest schema, loader, and content hash.

The manifest is the source of truth for what repositories belong to a workspace.
It is validated with Pydantic; a BLAKE3 hash of the canonical content is stored
on the ``:Workspace`` node so the indexer can detect manifest drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SupportedLanguage = Literal["python", "typescript", "javascript"]


class RepoConfig(BaseModel):
    """One repository inside a workspace. Either ``url`` or ``path`` must be set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1, max_length=128)]
    url: str | None = None
    path: str | None = None
    branch: str = "main"
    languages: list[SupportedLanguage] = Field(default_factory=lambda: ["python"])

    @model_validator(mode="after")
    def _exactly_one_source(self) -> RepoConfig:
        if (self.url is None) == (self.path is None):
            raise ValueError(
                f"repo '{self.name}' must specify exactly one of 'url' or 'path'"
            )
        return self


class WorkspaceConfig(BaseModel):
    """A workspace groups repos for cross-service edge resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    name: str
    repos: list[RepoConfig]

    @field_validator("repos")
    @classmethod
    def _unique_repo_names(cls, repos: list[RepoConfig]) -> list[RepoConfig]:
        seen: set[str] = set()
        for r in repos:
            if r.name in seen:
                raise ValueError(f"duplicate repo name '{r.name}' in workspace")
            seen.add(r.name)
        if not repos:
            raise ValueError("workspace must declare at least one repo")
        return repos


class Manifest(BaseModel):
    """Top-level manifest document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    workspaces: list[WorkspaceConfig]

    @field_validator("workspaces")
    @classmethod
    def _unique_workspace_ids(cls, workspaces: list[WorkspaceConfig]) -> list[WorkspaceConfig]:
        seen: set[str] = set()
        for w in workspaces:
            if w.id in seen:
                raise ValueError(f"duplicate workspace id '{w.id}'")
            seen.add(w.id)
        if not workspaces:
            raise ValueError("manifest must declare at least one workspace")
        return workspaces

    def workspace(self, workspace_id: str) -> WorkspaceConfig:
        for w in self.workspaces:
            if w.id == workspace_id:
                return w
        raise KeyError(f"workspace '{workspace_id}' not declared in manifest")


def load_manifest(path: Path) -> Manifest:
    """Parse and validate a YAML manifest file."""
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest root must be a mapping, got {type(raw).__name__}")
    return Manifest.model_validate(raw)


def manifest_sha(manifest: Manifest) -> str:
    """Stable content hash of a manifest (SHA-256, hex). Used for drift detection."""
    canonical = manifest.model_dump_json(round_trip=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
