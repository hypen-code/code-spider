"""Runtime configuration loaded from environment variables.

All settings can be overridden by `CODE_SPIDER_*` environment variables.
Use :func:`load_settings` at the entry point; pass the returned ``Settings`` down.

Two dotenv files are auto-loaded at import time (in order of decreasing
precedence — earlier wins, real env vars always win over both):

1. ``./.env`` at the current working directory (developer-friendly, project-local).
2. The user-global file at ``$CODE_SPIDER_CONFIG_FILE`` or, by default,
   ``$XDG_CONFIG_HOME/code-spider/config.env`` (falling back to
   ``~/.config/code-spider/config.env``).

The user-global path is what ``code-spider configure`` writes to, so that a
``pip install code-spider`` user can run ``code-spider serve`` from any
directory without copying a ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def user_config_path() -> Path:
    """Return the user-global config-env path (does not have to exist)."""
    override = os.environ.get("CODE_SPIDER_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "code-spider" / "config.env"


def _load_dotenv_layers() -> None:
    """Load CWD ``.env`` first (wins), then the user-global config file.

    ``override=False`` everywhere means an already-set real env var keeps
    winning, and the CWD ``.env`` wins over the user-global file (because it
    is loaded first).
    """
    load_dotenv(override=False)  # CWD/.env (and parent dirs via dotenv's search)
    user_path = user_config_path()
    if user_path.is_file():
        load_dotenv(dotenv_path=user_path, override=False)


_load_dotenv_layers()


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
class EmbeddingSettings:
    """Embedding-provider configuration sourced from ``CODE_SPIDER_EMBED_*``.

    ``provider`` chooses the backend (``sentence-transformers`` is the default
    local model; ``litellm`` routes through the LiteLLM SDK to any supported
    cloud provider; ``hash`` is the deterministic test/offline provider).
    All other fields are forwarded to the relevant adapter — most are only
    meaningful for ``litellm``.
    """

    provider: str
    model: str | None
    dim: int
    batch_size: int
    api_base: str | None
    api_key: str | None
    timeout_s: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class Settings:
    neo4j: Neo4jSettings
    embedding: EmbeddingSettings
    manifest_path: Path
    checkout_root: Path
    log_level: str
    log_json: bool


# ---- Embedding env loader ------------------------------------------------- #


_DEFAULT_EMBED_DIM = 384  # matches sentence-transformers/all-MiniLM-L6-v2


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"CODE_SPIDER_{name} must be an integer, got {raw!r}"
        ) from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"CODE_SPIDER_{name} must be a float, got {raw!r}"
        ) from exc


def _load_embedding_settings() -> EmbeddingSettings:
    """Read every ``CODE_SPIDER_EMBED_*`` knob with safe defaults."""
    raw_provider = _env("EMBED_PROVIDER", "sentence-transformers") or "sentence-transformers"
    return EmbeddingSettings(
        provider=raw_provider.strip(),
        model=_env("EMBED_MODEL"),
        dim=_env_int("EMBED_DIM", _DEFAULT_EMBED_DIM),
        batch_size=_env_int("EMBED_BATCH_SIZE", 64),
        api_base=_env("EMBED_API_BASE"),
        # Generic ``CODE_SPIDER_EMBED_API_KEY`` overrides; otherwise LiteLLM
        # itself picks the right provider-specific env var (OPENAI_API_KEY,
        # VOYAGE_API_KEY, COHERE_API_KEY, OPENROUTER_API_KEY, ...).
        api_key=_env("EMBED_API_KEY"),
        timeout_s=_env_float("EMBED_TIMEOUT_S", 30.0),
        max_retries=_env_int("EMBED_MAX_RETRIES", 3),
    )


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
        embedding=_load_embedding_settings(),
        manifest_path=Path(_env_required("MANIFEST_PATH", "./workspaces.yaml")).resolve(),
        checkout_root=Path(_env_required("CHECKOUT_ROOT", "./checkouts")).resolve(),
        log_level=_env_required("LOG_LEVEL", "INFO").upper(),
        log_json=_env("LOG_JSON", "0") in {"1", "true", "True", "yes"},
    )
