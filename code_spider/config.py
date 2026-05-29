"""Runtime configuration loaded from environment variables.

All settings can be overridden by `CODE_SPIDER_*` environment variables.
Use :func:`load_settings` at the entry point; pass the returned ``Settings`` down.

A ``.env`` file at the current working directory (or the project root above it)
is auto-loaded if present. Real environment variables always take precedence,
so CI/CD and one-off `CODE_SPIDER_FOO=... code-spider ...` invocations still
override the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load ``.env`` once at import time. ``override=False`` means an already-set
# real env var wins — important so CI secrets and ad-hoc overrides keep working.
load_dotenv(override=False)


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"CODE_SPIDER_{name}", default)


def _env_required(name: str, default: str | None = None) -> str:
    value = _env(name, default)
    if value is None or value == "":
        raise RuntimeError(
            f"Missing required environment variable CODE_SPIDER_{name}. "
            "Set it directly or create a .env file (see .env.example)."
        )
    return value


@dataclass(frozen=True, slots=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str


@dataclass(frozen=True, slots=True)
class Settings:
    neo4j: Neo4jSettings
    manifest_path: Path
    checkout_root: Path
    log_level: str
    log_json: bool


def load_settings() -> Settings:
    """Read configuration from environment, returning an immutable Settings."""
    neo4j = Neo4jSettings(
        uri=_env_required("NEO4J_URI", "bolt://localhost:7687"),
        user=_env_required("NEO4J_USER", "neo4j"),
        password=_env_required("NEO4J_PASSWORD", "codespider-dev-password"),
        database=_env_required("NEO4J_DATABASE", "neo4j"),
    )
    return Settings(
        neo4j=neo4j,
        manifest_path=Path(_env_required("MANIFEST_PATH", "./workspaces.yaml")).resolve(),
        checkout_root=Path(_env_required("CHECKOUT_ROOT", "./checkouts")).resolve(),
        log_level=_env_required("LOG_LEVEL", "INFO").upper(),
        log_json=_env("LOG_JSON", "0") in {"1", "true", "True", "yes"},
    )
