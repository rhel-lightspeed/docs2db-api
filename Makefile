.PHONY: test lint format clean help

help:
	@echo "Available commands:"
	@echo "  make test    - Run tests with pytest"
	@echo "  make lint    - Run linters (ruff, pyright)"
	@echo "  make format  - Format code with ruff"
	@echo "  make clean   - Remove generated files"

test:
	uv run pytest

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

