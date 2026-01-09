# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-01-09

### Added
- **Shell-friendly query output**: New `--format text` option for `query` command outputs clean document text without logs, suitable for shell scripts and LLM prompt injection
- **Context size limiting**: New `--max-chars` option truncates output to fit LLM token budgets

### Changed
- Query command now supports two output formats: `text` (clean) and `log` (verbose, default)

## [0.2.0] - 2025-11-24

### Added
- **Pydantic Settings**: Comprehensive configuration system with nested settings groups (`LLMSettings`, `DatabaseSettings`, `RAGSettings`, `LoggingSettings`)
  - Environment variable prefixes: `DOCS2DB_LLM_*`, `DOCS2DB_DB_*`, `DOCS2DB_RAG_*`, `DOCS2DB_LOG_LEVEL`
  - Type validation and coercion for all configuration values
  - `extra="ignore"` to safely coexist with other applications' environment variables
- **LLM Query Refinement Features**:
  - EMPTY response handling - skips RAG retrieval for non-technical questions (greetings, gibberish, etc.)
  - Auto-formatting cleanup for refined questions (removes numbering, quotes, markdown)
- **Cross-encoder Reranker Warmup**: Model loads and initializes during startup for faster first-request response and start-up time notification of a missing model.
- **Auto-initialization for LLM client**: When query refinement is enabled but no LLM client provided, automatically creates one from environment configuration
- **Comprehensive logging** throughout RAG pipeline with timing breakdowns and feature usage tracking

### Changed
- **Database configuration** now uses Pydantic settings instead of direct `os.getenv()` calls
  - New prefixed environment variables: `DOCS2DB_DB_*` (replaces `POSTGRES_*`)
  - Cleaner precedence hierarchy: Pydantic settings → `DOCS2DB_DB_URL` → compose file → defaults
- **Logging configuration** centralized in Pydantic settings (`DOCS2DB_LOG_LEVEL`)
- **RAG defaults** now sourced from Pydantic settings for consistency
- Improved error messages and debug logging throughout RAG engine
- Enhanced question refinement with format validation and auto-correction
- EMPTY response behavior now skips RAG retrieval

### Fixed
- PyTorch/torchvision version compatibility issues
- README async example corrections
- Type checking issues in tests with Optional environment variables

## [0.1.0] - 2025-11-12

### Added
- Universal RAG engine with hybrid search (vector + BM25)
- Reciprocal Rank Fusion (RRF) for combining search results
- Cross-encoder reranking for improved result quality
- Question refinement for better query expansion
- Multi-source database configuration with precedence hierarchy (CLI args → env vars → DATABASE_URL → compose file → defaults)
- RAG settings hierarchy system (query parameters → RAGConfig → database → defaults)
- Custom refinement prompt support for query expansion
- CLI commands: `db-status`, `db-start`, `db-stop`, `db-destroy`, `db-restore`, `manifest`, `query`
- LlamaStack integration for agent tool calling with demos
- Database utilities (`check_database_status`, `restore_database`, `generate_manifest`)
- Schema metadata and recent changes tracking in database
- Project URLs in package metadata (homepage, documentation, repository, issues, changelog)
- Keywords and classifiers in `pyproject.toml` for better PyPI discoverability

### Changed
- **BREAKING**: `UniversalRAGEngine` now uses two-phase initialization pattern (constructor + `await engine.start()`)
- **BREAKING**: RAGConfig fields now Optional (None = fall through to next level in hierarchy)
- `db-status` now displays document paths (without `/source.json` suffix) and shows schema metadata and recent changes
- Simplified `postgres-compose.yml` (removed adminer/pgadmin, standardized credentials to postgres/postgres)
- Completely rewritten README with quickstart guide, configuration hierarchy documentation, and improved structure
- Package description updated to better reflect functionality ("Query Docs2DB RAG databases with hybrid search and reranking")
- Improved type safety with assertions for Optional fields after initialization
- RAG engine now auto-detects embedding model from database if not specified
- Improved error messages for database connectivity issues and configuration conflicts

### Fixed
- Fixed pytest coverage configuration to use correct package name (`docs2db_api`)
- Improved error handling for database configuration conflicts (DATABASE_URL + POSTGRES_* vars)

## License

See [LICENSE](LICENSE) for details.

[Unreleased]: https://github.com/rhel-lightspeed/docs2db-api/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/rhel-lightspeed/docs2db-api/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rhel-lightspeed/docs2db-api/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rhel-lightspeed/docs2db-api/releases/tag/v0.1.0

