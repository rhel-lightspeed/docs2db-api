.PHONY: test test-ci test-integration lint format clean help

help:
	@echo "Available commands:"
	@echo "  make test             - Run all tests with pytest"
	@echo "  make test-ci          - Run CI-safe tests (excludes no_ci marker)"
	@echo "  make test-integration - Run integration tests (requires PostgreSQL on port 5433)"
	@echo "  make lint             - Run linters (ruff, pyright)"
	@echo "  make format           - Format code with ruff"
	@echo "  make clean            - Remove generated files"

test:
	uv run pytest

test-ci:
	uv run pytest -m "not no_ci"

test-integration:
	uv run pytest -m "integration"

lint:
	uv run ruff check .
	uv run pyright

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov coverage.xml .coverage
