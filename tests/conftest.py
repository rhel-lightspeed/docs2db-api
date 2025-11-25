"""Pytest fixtures and configuration for docs2db-api tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


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
    client.refine_query = AsyncMock(return_value=[
        "What is X?",
        "How does X work?",
        "Where is X used?",
    ])
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

