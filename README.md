# Code Spider

Centralized codebase knowledge graph + coordinate index for AI coding agents.
Backed by **Neo4j 5.x Community**, written in **Python 3.13+**, parses with **Tree-sitter**,
exposes the graph to agents via the **Model Context Protocol (MCP)**.

> Status: **Phase 0 — Foundations**. End-to-end indexing for a single Python repo into Neo4j is the current goal.
> Phases 1 (TS/JS, REST flow, Kafka flow, MCP server, hybrid search) and 2 (incremental, observability) follow.

## Why

AI coding agents waste enormous context windows on grep/list/read loops while exploring large
polyglot codebases. Code Spider precomputes the structural + semantic shape of an entire workspace
(every symbol, import, call, REST route, Kafka topic flow, code chunk embedding) into a single
queryable Neo4j graph, then exposes navigation primitives via MCP so agents can:

- Jump directly to file/line coordinates without scanning.
- Trace call graphs, impact analysis, and cross-service HTTP/Kafka flows in a single Cypher hop.
- Resolve natural-language queries via hybrid lexical + vector search and receive precise coordinates.

See the design plan: `~/.windsurf/plans/code-spider-knowledge-graph-aea777.md`.

## Architecture (one screen)

```
workspaces.yaml --> CI indexer ----> Neo4j 5.x Community
                       |                  ^
                       v                  | Cypher
                Shared FS (commit SHA)    |
                       ^                  |
                       +----- MCP server (Python)
                                          ^
                                          | MCP / JSON-RPC
                                  AI agents (Windsurf / Cursor / Claude Code / Codex)
```

## Locked design decisions

| Dimension | Decision |
|---|---|
| Topology | Single shared central Neo4j 5.x Community |
| MVP languages | Python, TypeScript, JavaScript |
| Cross-service edges | REST/HTTP + Kafka producer/consumer |
| Enrichment | Structural + hybrid lexical/vector search (RRF) |
| Indexing trigger | CI pipeline step on merge to main |
| Vector storage | Neo4j native HNSW (abstracted behind `VectorBackend`) |
| Call resolution | Tree-sitter + 6-strategy heuristic cascade |
| Agent interface | MCP server only |
| Workspace model | Explicit `workspaces.yaml` manifest |
| Embedding model | Local `sentence-transformers` by default; optional LiteLLM-backed external models (Voyage, OpenAI, Cohere, OpenRouter) via `.env` |
| Snippet retrieval | Indexer-managed shared filesystem keyed by commit SHA |

## Quickstart for developers (consume an existing central graph)

If your team already runs a central Neo4j with the graph indexed, this is all
you need. No Docker, no local Neo4j, no indexing.

```bash
# 1. Install (requires Python 3.12+)
pip install code-spider              # or: pipx install code-spider
# or zero-install with uv:           uvx code-spider serve

# 2. Point it at the central Neo4j
code-spider configure                # interactive wizard, saves to
                                     # ~/.config/code-spider/config.env (0600)

# 3. Verify the connection end-to-end
code-spider doctor                   # checks env -> bolt -> auth -> schema

# 4. Print the MCP JSON snippet for your coding agent
code-spider mcp-config --agent windsurf       # or: cursor | claude-code | generic
# Paste the printed JSON into the path the wizard tells you about.
```

That's it — restart your agent and the `code-spider` MCP server is wired in.

### Supported coding agents

| Agent | Where to paste the `mcp-config` output |
|---|---|
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| Cursor | `~/.cursor/mcp.json` (or project-level `.cursor/mcp.json`) |
| Claude Code | `claude mcp add-json code-spider '<inner object>'` |
| Generic | Any MCP client that consumes the standard JSON schema |

## Quickstart for admins (run the central server)

This is the side that operates Neo4j, defines `workspaces.yaml`, and indexes
repos in CI on every merge to `main`.

### 1. Start Neo4j Community

```bash
docker compose up -d neo4j
# Browser: http://localhost:7474  (neo4j / codespider-dev-password)
```

### 2. Install with dev extras

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,embedding]"
```

### 3. Deploy graph schema

```bash
code-spider migrate
```

### 4. Index repositories

```bash
cp workspaces.example.yaml workspaces.yaml
# edit workspaces.yaml to point at real repos (path or git URL)
code-spider index --workspace demo
```

### 5. Verify

```cypher
// in Neo4j Browser
MATCH (s:Symbol) RETURN s.kind, count(*) AS n ORDER BY n DESC;
```

### 6. Production indexing options

```bash
# Full run with embeddings + Prometheus metrics
code-spider index --workspace demo --embed sentence-transformers --metrics-port 9464

# Incremental on subsequent CI runs (skip unchanged files)
code-spider index --workspace demo --incremental --embed auto

# Prometheus scraping
curl http://localhost:9464/metrics | grep code_spider_
```

### 6a. External embedding models (LiteLLM)

The default `sentence-transformers/all-MiniLM-L6-v2` runs locally and needs no
API key. For production-grade code retrieval quality you can switch to a
hosted model via the [LiteLLM](https://docs.litellm.ai/) SDK without touching
any code:

```bash
pip install -e ".[litellm]"        # adds the litellm dependency
```

Pick one of the recommended models in `.env`:

| Model | Dim | Strengths | Env vars |
|---|---|---|---|
| **`voyage/voyage-code-3`** *(recommended for code)* | 1024 | Tuned on source code; tops code-retrieval benchmarks | `VOYAGE_API_KEY` |
| `openai/text-embedding-3-small` | 1536 | Cheap, widely available, strong general baseline | `OPENAI_API_KEY` |
| `cohere/embed-multilingual-v3.0` | 1024 | Multilingual code + prose | `COHERE_API_KEY` |
| OpenRouter (OpenAI-compatible) | varies | Single key, many backends *(verify the chosen route exposes /embeddings)* | `CODE_SPIDER_EMBED_API_BASE`, `CODE_SPIDER_EMBED_API_KEY` |

`.env` example for Voyage:

```dotenv
CODE_SPIDER_EMBED_PROVIDER=litellm
CODE_SPIDER_EMBED_MODEL=voyage/voyage-code-3
CODE_SPIDER_EMBED_DIM=1024
VOYAGE_API_KEY=...
```

Then re-create the vector index at the new dimension and reindex:

```bash
code-spider migrate                                       # auto-recreates index at CODE_SPIDER_EMBED_DIM
code-spider index --workspace demo                        # picks up litellm via .env
```

`migrate` auto-detects when `CODE_SPIDER_EMBED_DIM` differs from the
existing `chunk_embedding` index and drops + recreates the index at the
new dimension. **This deletes every existing chunk embedding**, so you
must reindex affected workspaces afterwards (you would need to anyway —
vectors from one model can't be compared to vectors from another).

Precedence: an explicit `--embed <name>` flag always wins; `--embed auto`
(the default) reads `CODE_SPIDER_EMBED_PROVIDER`.

### 7. Recommended security model for developers

Create a read-only Neo4j user for developers so a leaked password can't
mutate the graph:

```cypher
// run as the admin user in Neo4j Browser
CREATE USER codespider_ro SET PASSWORD 'rotate-me' CHANGE NOT REQUIRED;
GRANT ROLE reader TO codespider_ro;
```

Hand `codespider_ro` (not the admin user) to developers running
`code-spider configure`.

### 8. Hand-rolled MCP JSON (if you don't want to use `mcp-config`)

```json
{
  "mcpServers": {
    "code-spider": {
      "command": "/absolute/path/to/code-spider",
      "args": ["serve"],
      "env": {
        "CODE_SPIDER_NEO4J_URI": "bolt://central-neo4j.example.com:7687",
        "CODE_SPIDER_NEO4J_USER": "codespider_ro",
        "CODE_SPIDER_NEO4J_PASSWORD": "rotate-me",
        "CODE_SPIDER_NEO4J_DATABASE": "neo4j"
      }
    }
  }
}
```

## Layout

```
code_spider/
├── config.py             # env + manifest loading (CWD .env + ~/.config/code-spider/config.env)
├── onboarding.py         # `configure` wizard, `mcp-config`, `doctor`
├── workspace/manifest.py # YAML schema + diff
├── checkout/git.py       # GitPython wrapper
├── parser/               # tree-sitter language adapters
├── symbols/              # domain model + FQN helpers
├── resolver/             # 6-strategy cascade (Phase 1)
├── routes/               # REST extractors + HTTP_FLOW matcher (Phase 1)
├── messaging/            # Kafka extractors + KAFKA_FLOW matcher (Phase 1)
├── chunker/              # AST-aware chunker (Phase 1)
├── embedding/            # sentence-transformers wrapper (Phase 1)
├── graph/                # Neo4j client, schema, writer, vector backends
├── search/               # lexical + vector + RRF fusion (Phase 1)
├── mcp/                  # MCP server + 8 tools (Phase 1)
└── cli.py                # `code-spider configure|doctor|mcp-config|migrate|index|serve`
```

## Development

```bash
pytest                                # unit tests
pytest -m integration                 # requires Neo4j on localhost:7687
ruff check . && ruff format --check . # lint + format
mypy code_spider                      # type-check
```

## License

Apache-2.0
