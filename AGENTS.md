# AGENTS.md

Knowledge base for AI agents working on this repository.

## Project Overview

docs2db-api is a Typer CLI library for RAG retrieval from databases built by docs2db.

**Architecture:** Typer CLI → async psycopg v3 → PostgreSQL/pgvector → hybrid BM25+vector search → cross-encoder reranking → optional LLM query refinement

**Related repositories:**

- [docs2db](https://github.com/rhel-lightspeed/docs2db) — builds the RAG databases this server queries
- [docs2db-mcp-server](https://github.com/rhel-lightspeed/docs2db-mcp-server) — MCP interface for AI assistants

**Author:** Ellis Low (elow@redhat.com)
**License:** Apache-2.0
**Python:** 3.12 (strict — no other versions)
**Package manager:** uv
**CLI framework:** Typer

## Development Environment

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Start the database (requires Docker/Podman)
uv run docs2db-api db-start

# Run tests (requires PostgreSQL on port 5433)
make test

# Run linting/formatting/type checks
uv run pre-commit run --all-files

# CI-safe test run (excludes tests needing external services)
make test-ci
```

## Code Architecture

```text
src/docs2db_api/
├── docs2db_api.py   # FastAPI app + Typer CLI (315 lines) — HTTP endpoints and CLI commands
├── database.py      # Async PostgreSQL + pgvector operations (980 lines) — largest data layer
├── rag/
│   ├── engine.py       # Universal RAG engine — hybrid search, RRF, reranking (1043 lines)
│   └── llama_stack.py  # LlamaStack agent tool integration (460 lines)
├── config.py        # Pydantic settings with DOCS2DB_* env vars (112 lines)
├── embeddings.py    # Embedding model configs (Granite, E5, etc.) (122 lines)
├── reranker.py      # Cross-encoder reranker wrapper (96 lines)
├── db_lifecycle.py  # Database start/stop/destroy via Docker/Podman (324 lines)
├── exceptions.py    # Custom exception hierarchy (32 lines)
└── __init__.py      # Package init (26 lines)
```

## Key Patterns and Conventions

- **Logging:** structlog (`structlog.get_logger()`). Do NOT use stdlib `logging`.
- **Log formatting:** Avoid f-strings in structlog calls. Use `%s` style for lazy evaluation.
- **Config/settings:** Pydantic settings (`pydantic-settings`). All env vars use `DOCS2DB_*` prefix. Nested groups: `DOCS2DB_DB_*`, `DOCS2DB_LLM_*`, `DOCS2DB_RAG_*`, `DOCS2DB_LOG_LEVEL`.
- **PostgreSQL driver:** psycopg v3 ASYNC (`psycopg[binary]`). NOT psycopg2. NOT sync psycopg.
- **RAG initialization:** Two-phase pattern — `UniversalRAGEngine()` constructor + `await engine.start()`. The `start()` call loads the embedding model and warms up the reranker. Never skip the second phase.
- **PyTorch:** CPU-only by design. `pyproject.toml` uses the `pytorch-cpu` index. Do not add GPU paths.
- **Async:** FastAPI lifespan events manage the engine lifecycle. All database calls must be `async`.

## Testing

- **Framework:** pytest with pytest-asyncio, pytest-cov, pytest-httpx, pytest-randomly
- **Tests are mock-based.** External services (PostgreSQL, embeddings) are mocked in tests.
- **Test database:** Port 5433 (NOT 5432), user `test_user`, password `test_password`, database `test_docs2db`
- **Coverage:** Configured in `pyproject.toml` (term-missing, html, xml, branch)
- **Markers:** `no_ci` — tests requiring external services (Podman, real PostgreSQL, etc.)
- **CI test command:** `make test-ci` (runs `pytest -m "not no_ci"`)

## Pre-commit Hooks

These run on every commit and in CI:

- **ruff** — linting with auto-fix
- **ruff-format** — code formatting
- **pyright** — type checking (src/docs2db_api/ only)
- **gitleaks** — secret detection (src/ and tests/ only)
- **check-toml** — TOML validation
- **end-of-file-fixer** — ensures files end with newline
- **trailing-whitespace** — removes trailing spaces

Run manually: `uv run pre-commit run --all-files`

## Gotchas

- psycopg v3 is ASYNC — all `execute()`, `fetchall()`, etc. must be `await`ed
- RAG engine requires two-phase init: `engine = UniversalRAGEngine(...)` then `await engine.start()`
- Test DB runs on port 5433 (NOT 5432) with separate credentials
- `make test-ci` excludes `no_ci` marked tests
- torch is CPU-only by design (pyproject.toml uses pytorch-cpu index)
- Embedding model loading takes 30-40s on first load — cold starts are slow
- f-strings in structlog calls are discouraged (use `%s` style for lazy evaluation)
- `rag/engine.py` is 1043 lines — the largest and most complex module

## Branch Protection

- **Org-level ruleset:** "Minimum required Branch Protection" (rhel-lightspeed org)
- Requires 1 approving review from code owner (`@rhel-lightspeed/developers`)
- Last pusher cannot approve their own PR
- Cannot be bypassed

## Changelog Policy

Every PR must include an update to `CHANGELOG.md` under the `## [Unreleased]` section.

Follow the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format:

- **Added** — new features
- **Changed** — changes in existing functionality
- **Deprecated** — soon-to-be removed features
- **Removed** — removed features
- **Fixed** — bug fixes
- **Security** — vulnerability fixes

Keep entries concise (1-2 lines each). Reference issue numbers where applicable.
