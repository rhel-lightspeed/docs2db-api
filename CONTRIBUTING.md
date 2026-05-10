# Contributing to docs2db-api

Thank you for your interest in contributing to docs2db-api! This guide covers development setup and workflow.

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- Podman or Docker — for running PostgreSQL

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rhel-lightspeed/docs2db-api
   cd docs2db-api
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

## Running the API Server

Start PostgreSQL first, then the FastAPI server:

```bash
uv run docs2db-api db-start
uv run uvicorn docs2db_api.docs2db_api:api_app --reload
```

## CLI Commands

```bash
uv run docs2db-api db-start     # Start PostgreSQL container
uv run docs2db-api db-stop      # Stop container (data persists)
uv run docs2db-api db-destroy   # Delete database and volumes
uv run docs2db-api db-status    # Check connection and stats
uv run docs2db-api query "your question here"
```

## Testing

Run tests (requires PostgreSQL on port 5433):

```bash
make test
```

Run specific tests:

```bash
uv run pytest tests/test_rag_config.py
uv run pytest tests/test_rag_config.py::test_specific_function
```

CI-safe run (skips tests marked `no_ci`):

```bash
make test-ci
```

## Code Quality

Pre-commit hooks run automatically on `git commit`:

```bash
uv run pre-commit run --all-files
```

This runs:
- **ruff** — linting with auto-fixes
- **ruff-format** — code formatting
- **pyright** — type checking (`src/docs2db_api/` only)
- **gitleaks** — secret detection
- **check-toml** — TOML file validation
- **end-of-file-fixer** — ensures files end with newline
- **trailing-whitespace** — removes trailing spaces

## Continuous Integration

Pull requests are automatically checked by GitHub Actions:

- **Lint**: ruff (linting + formatting) and pyright (type checking)
- **Test**: Full test suite against PostgreSQL (port 5433)

Both checks must pass before merge.

## Branching

Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Every PR must include an update to `CHANGELOG.md` under `## [Unreleased]`.

## Code Style

- **Python version:** 3.12
- **Formatter:** Ruff (enforced by pre-commit)
- **Type hints:** Required for public APIs
- **Imports:** Sorted by Ruff (isort rules)
- **Logging:** structlog only — do NOT use stdlib `logging`

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
