# docs2db-api Tests

This directory contains automated tests for `docs2db-api`. The test suite is designed to validate the RAG settings hierarchy and configuration logic **without requiring PostgreSQL**.

## Test Philosophy

The tests focus on **unit testing the configuration logic and data structures** rather than integration testing the full database-connected pipeline. This approach offers:

- ✅ **Fast execution** - No database setup required
- ✅ **Reliable in CI** - No external dependencies
- ✅ **Easy to run locally** - Just `make test` or `uv run pytest`
- ✅ **Clear failures** - Tests are focused and easy to debug

## What's Tested

### Configuration (`test_rag_config.py`)

1. **RAGConfig Dataclass**
   - Default values (all fields None for settings hierarchy)
   - Explicit value assignment
   - Partial configuration (some values set, others None)

2. **Default Settings Constants**
   - `DEFAULT_RAG_SETTINGS` contains all required keys
   - Default values are reasonable and valid
   - Boolean features default to enabled

3. **Environment Variable Parsing**
   - Float parsing (e.g., `SIMILARITY_THRESHOLD=0.88`)
   - Integer parsing (e.g., `MAX_CHUNKS=25`)
   - Boolean parsing (various formats: true/false, 1/0, yes/no)
   - Handling of unset variables

4. **Settings Hierarchy Logic**
   - CLI/kwargs override everything
   - Database values override defaults
   - Defaults used when nothing else set
   - Edge cases: 0 and False are valid explicit values

5. **RAGResult Structure**
   - Can be created with required fields
   - Has correct attributes

## What's NOT Tested

Integration tests requiring PostgreSQL are **intentionally excluded** to keep the test suite fast and dependency-free. These would include:

- Database connection and initialization
- Model auto-detection from database
- Reading `rag_settings` table
- Full end-to-end search pipeline
- Actual embedding generation
- Cross-encoder reranking with real models

For comprehensive integration testing, use the PostgreSQL-backed tests in `docs2db` (the parent project), or run manual integration tests locally.

## Running Tests

```bash
# Run all tests
make test

# Run tests with verbose output
uv run pytest -v

# Run specific test class
uv run pytest tests/test_rag_config.py::TestRAGConfig -v

# Run specific test
uv run pytest tests/test_rag_config.py::TestRAGConfig::test_config_defaults_are_none -v

# Run with coverage
uv run pytest --cov
```

## Coverage

Current test coverage focuses on the **new RAG settings hierarchy feature** added in version 0.2.0:
- Configuration dataclasses
- Default settings
- Settings precedence logic
- Environment variable handling

Coverage of database-connected code is minimal by design, as those components require PostgreSQL for meaningful testing.

## Adding Tests

When adding new tests:

1. **Keep them unit-focused** - Test individual functions/classes
2. **Avoid database dependencies** - Use mocks sparingly, prefer testing logic directly
3. **Test edge cases** - Especially for settings (0, False, None handling)
4. **Document what you're testing** - Clear test names and docstrings

## Future Enhancements

Potential additions for the test suite:

- Mark some tests with `@pytest.mark.no_ci` for optional PostgreSQL integration tests
- Add tests for custom refinement prompts (when feature is more mature)
- Add tests for search options override behavior
- Consider snapshot testing for query results (once stable)

