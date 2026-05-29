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

#: Default ceiling for per-input embedding text in characters. Tuned just
#: below the 131_072-char cap shared by Voyage / Qwen-3 / OpenAI's larger
#: context models so we leave ~10 KB of safety overhead.
_DEFAULT_MAX_INPUT_CHARS = 120_000

#: Default ceiling for files we even attempt to parse / chunk / embed.
#: Files bigger than this are almost always auto-generated assets (minified
#: bundles, vendored libraries, lockfiles, JSON dumps, package manifests)
#: whose semantic value for code intelligence is near zero, and whose chunks
#: blow up the embedding bill and memory budget. 1 MiB strikes a balance:
#: long enough to keep all real source files; short enough to keep the
#: indexer well under 4 GiB peak resident on a typical small CI runner.
_DEFAULT_MAX_FILE_BYTES = 1_048_576  # 1 MiB

#: Default wall-clock ceiling (seconds) for a single MCP tool invocation.
#: A tool that runs longer than this is aborted so an agent never waits
#: indefinitely on a slow/hung query. Override with
#: ``CODE_SPIDER_TOOL_TIMEOUT_S``; set ``0`` (or negative) to disable.
_DEFAULT_TOOL_TIMEOUT_S = 20.0

#: Default wall-clock ceiling (seconds) for ``index_repository``. Indexing a
#: workspace clones repos, parses every file and embeds chunks — it routinely
#: runs for minutes, far longer than the generic tool timeout. We therefore
#: give it its own knob (``CODE_SPIDER_INDEX_TIMEOUT_S``) and disable the
#: timeout by default (``0``) so an interactive reindex is never killed
#: mid-run. Set a positive value to cap it.
_DEFAULT_INDEX_TIMEOUT_S = 0.0


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
    # Per-input character cap. Most hosted embedding models reject any single
    # input longer than ~131_072 characters (Qwen-3, Voyage, OpenAI's larger
    # context models all sit in that range). We default to 120_000 to leave
    # headroom and pre-truncate anything longer at the provider boundary so a
    # single huge auto-generated or minified file can't crash the whole
    # workspace embed. Override with ``CODE_SPIDER_EMBED_MAX_INPUT_CHARS``.
    max_input_chars: int
    # Number of concurrent embedding sub-batches dispatched per repo. The
    # embedding stage is I/O-bound (network calls to the provider), so a
    # thread pool — not a process pool — is the right primitive. The default
    # is ``min(os.cpu_count() or 2, 4)``: enough parallelism to saturate a
    # modest upstream rate budget without thrashing free-tier providers, and
    # well-sized for the 2 vCPU / 4 GiB target box. Override with
    # ``CODE_SPIDER_EMBED_WORKERS``.
    workers: int


@dataclass(frozen=True, slots=True)
class Settings:
    neo4j: Neo4jSettings
    embedding: EmbeddingSettings
    manifest_path: Path
    checkout_root: Path
    log_level: str
    log_json: bool
    # Hard ceiling on file size considered for parse/chunk/embed. Files
    # larger than this are skipped at the walker with a counter + warning.
    # See ``_DEFAULT_MAX_FILE_BYTES`` for the rationale.
    max_file_bytes: int
    # Wall-clock ceiling (seconds) for a single MCP tool invocation. A tool
    # exceeding this is aborted with a ``TimeoutError`` so agents never wait
    # indefinitely. ``<= 0`` disables the timeout. See
    # ``_DEFAULT_TOOL_TIMEOUT_S``.
    tool_timeout_s: float
    # Wall-clock ceiling (seconds) specific to ``index_repository``, which is a
    # long-running (minutes) operation and must not be governed by the generic
    # ``tool_timeout_s``. ``<= 0`` disables the timeout. See
    # ``_DEFAULT_INDEX_TIMEOUT_S``.
    index_timeout_s: float


# ---- Embedding env loader ------------------------------------------------- #


_DEFAULT_EMBED_DIM = 384  # matches sentence-transformers/all-MiniLM-L6-v2


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"CODE_SPIDER_{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"CODE_SPIDER_{name} must be a float, got {raw!r}") from exc


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
        max_input_chars=_env_int("EMBED_MAX_INPUT_CHARS", _DEFAULT_MAX_INPUT_CHARS),
        workers=_env_int("EMBED_WORKERS", _default_workers()),
    )


def _default_workers() -> int:
    """Default embedding concurrency: ``min(cpu_count, 4)`` (min 1).

    Capped at 4 so we don't hammer free-tier embedding endpoints on bigger
    boxes; tunable via ``CODE_SPIDER_EMBED_WORKERS``. We pick 4 (not 8) as
    the upper bound because the 2-vCPU / 4 GiB target box can't usefully
    parallelise further, and most free-tier API quotas top out around there.
    """
    cpu = os.cpu_count() or 2
    return max(1, min(cpu, 4))


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
        max_file_bytes=_env_int("MAX_FILE_BYTES", _DEFAULT_MAX_FILE_BYTES),
        tool_timeout_s=_env_float("TOOL_TIMEOUT_S", _DEFAULT_TOOL_TIMEOUT_S),
        index_timeout_s=_env_float("INDEX_TIMEOUT_S", _DEFAULT_INDEX_TIMEOUT_S),
    )
