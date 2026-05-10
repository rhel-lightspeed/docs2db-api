"""Pytest fixtures and configuration for docs2db-api tests."""

import os
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest

_TEST_DB_CONFIG = {
    "host": os.getenv("TEST_DB_HOST", "localhost"),
    "port": int(os.getenv("TEST_DB_PORT", "5433")),
    "database": os.getenv("TEST_DB_NAME", "test_docs2db"),
    "user": os.getenv("TEST_DB_USER", "test_user"),
    "password": os.getenv("TEST_DB_PASSWORD", "test_password"),
}


@pytest.fixture(scope="session")
def pg_available() -> bool:
    """Session-scoped check: is the test PostgreSQL instance reachable?"""
    cfg = _TEST_DB_CONFIG
    conn_string = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    try:
        with psycopg.Connection.connect(conn_string, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture
def test_db_config() -> dict:
    """Test database configuration dictionary."""
    return _TEST_DB_CONFIG.copy()


@pytest.fixture
def skip_if_no_pg(pg_available: bool) -> None:
    """Skip the test if PostgreSQL is not available on port 5433."""
    if not pg_available:
        pytest.skip("PostgreSQL not available on port 5433 — skipping integration test")


@pytest.fixture
async def pg_connection(pg_available: bool, test_db_config: dict):
    """Async psycopg3 connection to test PostgreSQL; skips if unavailable."""
    if not pg_available:
        pytest.skip("PostgreSQL not available on port 5433 — skipping integration test")

    cfg = test_db_config
    conn_string = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    conn = await psycopg.AsyncConnection.connect(conn_string)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager for testing without PostgreSQL."""
    manager = AsyncMock()

    # Mock connection context manager
    conn_mock = AsyncMock()
    conn_mock.__aenter__ = AsyncMock(return_value=conn_mock)
    conn_mock.__aexit__ = AsyncMock(return_value=None)

    manager.get_direct_connection.return_value = conn_mock

    return manager


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for testing."""
    provider = MagicMock()
    provider.encode.return_value = [[0.1, 0.2, 0.3] * 128]  # Mock 384-dim embedding
    return provider


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for query refinement testing."""
    client = MagicMock()
    client.acomplete = AsyncMock(
        return_value="1. What is X?\n2. How does X work?\n3. Where is X used?"
    )
    return client


@pytest.fixture
def sample_rag_settings():
    """Sample RAG settings from database."""
    return {
        "refinement_prompt": "Test prompt with {question}",
        "enable_refinement": True,
        "enable_reranking": True,
        "similarity_threshold": 0.8,
        "max_chunks": 15,
        "max_tokens_in_context": 8192,
        "refinement_questions_count": 5,
    }


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "chunk_id": 1,
            "document_id": 1,
            "text": "Sample chunk 1",
            "document_path": "doc1.md",
            "similarity_score": 0.95,
        },
        {
            "chunk_id": 2,
            "document_id": 2,
            "text": "Sample chunk 2",
            "document_path": "doc2.md",
            "similarity_score": 0.85,
        },
    ]
