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
| Embedding model | Local `sentence-transformers` in-process |
| Snippet retrieval | Indexer-managed shared filesystem keyed by commit SHA |

## Quickstart (Phase 0)

### 1. Start a local Neo4j Community

```bash
docker compose up -d neo4j
# Browser: http://localhost:7474  (neo4j / codespider-dev-password)
```

### 2. Install in editable mode

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Deploy graph schema

```bash
code-spider migrate
```

### 4. Index a Python repo

```bash
cp workspaces.example.yaml workspaces.yaml
# edit workspaces.yaml to point at a real repo (path or git URL)
code-spider index --workspace demo
```

### 5. Verify

```cypher
// in Neo4j Browser
MATCH (s:Symbol) RETURN s.kind, count(*) AS n ORDER BY n DESC;
```

### 6. How to drive it
```bash
# Full Phase 1 indexing run with embeddings + metrics
code-spider migrate
code-spider index --workspace demo --embed sentence-transformers --metrics-port 9464

# Phase 2 incremental on subsequent runs (skip unchanged files)
code-spider index --workspace demo --incremental --embed auto

# MCP server (stdio JSON-RPC) for agent consumption
code-spider serve

# Prometheus metrics (when --metrics-port supplied)
curl http://localhost:9464/metrics | grep code_spider_
```

### 7. MCP server

```json
{
  "mcpServers": {
    "code-spider": {
      "command": "~/code-spider/.venv/bin/code-spider",
      "args": ["serve", "--embed", "auto"],
      "cwd": "~/code-spider",
      "env": {
        "CODE_SPIDER_NEO4J_URI": "bolt://localhost:7687",
        "CODE_SPIDER_NEO4J_USER": "neo4j",
        "CODE_SPIDER_NEO4J_PASSWORD": "password",
        "CODE_SPIDER_NEO4J_DATABASE": "neo4j",
        "CODE_SPIDER_MANIFEST_PATH": "~/code-spider/workspaces.yaml",
        "CODE_SPIDER_CHECKOUT_ROOT": "~/code-spider/checkouts",
        "CODE_SPIDER_LOG_LEVEL": "INFO",
        "CODE_SPIDER_LOG_JSON": "1"
      }
    }
  }
}
```

## Layout

```
code_spider/
├── config.py             # env + manifest loading
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
└── cli.py                # `code-spider migrate|index|serve`
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
