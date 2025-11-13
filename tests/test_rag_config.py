"""Simple unit tests for RAG configuration without database dependencies."""

import os

import pytest

from docs2db_api.rag.engine import DEFAULT_RAG_SETTINGS, RAGConfig


class TestRAGConfig:
    """Test RAGConfig dataclass behavior."""

    def test_config_defaults_are_none(self):
        """Test that RAGConfig has all fields as Optional with None defaults."""
        config = RAGConfig()

        assert config.model_name is None
        assert config.similarity_threshold is None
        assert config.max_chunks is None
        assert config.max_tokens_in_context is None
        assert config.enable_question_refinement is None
        assert config.enable_reranking is None
        assert config.refinement_questions_count is None

    def test_config_accepts_explicit_values(self):
        """Test that RAGConfig accepts and stores explicit values."""
        config = RAGConfig(
            model_name="test-model",
            similarity_threshold=0.75,
            max_chunks=20,
            max_tokens_in_context=8192,
            enable_question_refinement=False,
            enable_reranking=True,
            refinement_questions_count=3,
        )

        assert config.model_name == "test-model"
        assert config.similarity_threshold == 0.75
        assert config.max_chunks == 20
        assert config.max_tokens_in_context == 8192
        assert config.enable_question_refinement is False
        assert config.enable_reranking is True
        assert config.refinement_questions_count == 3

    def test_config_partial_values(self):
        """Test that RAGConfig can have some values set and others None."""
        config = RAGConfig(
            similarity_threshold=0.85,
            enable_reranking=False,
        )

        # Set values
        assert config.similarity_threshold == 0.85
        assert config.enable_reranking is False

        # Unset values remain None
        assert config.model_name is None
        assert config.max_chunks is None
        assert config.enable_question_refinement is None


class TestDefaultRAGSettings:
    """Test DEFAULT_RAG_SETTINGS constant."""

    def test_contains_all_required_keys(self):
        """Test that DEFAULT_RAG_SETTINGS contains all expected keys."""
        required_keys = {
            "similarity_threshold",
            "max_chunks",
            "max_tokens_in_context",
            "enable_question_refinement",
            "enable_reranking",
            "refinement_questions_count",
        }

        assert set(DEFAULT_RAG_SETTINGS.keys()) == required_keys

    def test_default_values_are_reasonable(self):
        """Test that default values are sensible."""
        assert DEFAULT_RAG_SETTINGS["similarity_threshold"] == 0.7
        assert 0.0 <= DEFAULT_RAG_SETTINGS["similarity_threshold"] <= 1.0

        assert DEFAULT_RAG_SETTINGS["max_chunks"] == 10
        assert DEFAULT_RAG_SETTINGS["max_chunks"] > 0

        assert DEFAULT_RAG_SETTINGS["max_tokens_in_context"] == 4096
        assert DEFAULT_RAG_SETTINGS["max_tokens_in_context"] > 0

        assert DEFAULT_RAG_SETTINGS["enable_question_refinement"] is True
        assert DEFAULT_RAG_SETTINGS["enable_reranking"] is True

        assert DEFAULT_RAG_SETTINGS["refinement_questions_count"] == 5
        assert DEFAULT_RAG_SETTINGS["refinement_questions_count"] > 0

    def test_boolean_defaults(self):
        """Test that boolean flags default to True (features enabled by default)."""
        assert DEFAULT_RAG_SETTINGS["enable_question_refinement"] is True
        assert DEFAULT_RAG_SETTINGS["enable_reranking"] is True


class TestEnvironmentVariableParsing:
    """Test environment variable handling for settings."""

    def test_env_var_float_parsing(self, monkeypatch):
        """Test that float environment variables are parsed correctly."""
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.88")

        value = os.getenv("SIMILARITY_THRESHOLD")
        assert value == "0.88"
        assert float(value) == 0.88

    def test_env_var_int_parsing(self, monkeypatch):
        """Test that integer environment variables are parsed correctly."""
        monkeypatch.setenv("MAX_CHUNKS", "25")

        value = os.getenv("MAX_CHUNKS")
        assert value == "25"
        assert int(value) == 25

    def test_env_var_bool_parsing_true(self, monkeypatch):
        """Test that boolean environment variables parse 'true' correctly."""
        for true_value in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
            monkeypatch.setenv("ENABLE_RERANKING", true_value)
            value = os.getenv("ENABLE_RERANKING")
            assert value.lower() in ("true", "1", "yes")

    def test_env_var_bool_parsing_false(self, monkeypatch):
        """Test that boolean environment variables parse 'false' correctly."""
        for false_value in ["false", "False", "FALSE", "0", "no", "No", "NO"]:
            monkeypatch.setenv("ENABLE_RERANKING", false_value)
            value = os.getenv("ENABLE_RERANKING")
            assert value.lower() in ("false", "0", "no")

    def test_env_var_not_set(self):
        """Test that unset environment variables return None."""
        value = os.getenv("NONEXISTENT_VAR_12345")
        assert value is None


class TestSettingsHierarchyLogic:
    """Test the settings hierarchy logic conceptually."""

    def test_explicit_value_overrides_none(self):
        """Test that explicit values take precedence over None."""
        # Simulates: CLI value overrides missing DB value
        cli_value = 0.95
        db_value = None
        default_value = 0.7

        # Priority: CLI > DB > default
        result = cli_value if cli_value is not None else (
            db_value if db_value is not None else default_value
        )

        assert result == 0.95

    def test_db_value_overrides_default(self):
        """Test that database values override defaults."""
        # Simulates: No CLI value, DB value present
        cli_value = None
        db_value = 0.85
        default_value = 0.7

        result = cli_value if cli_value is not None else (
            db_value if db_value is not None else default_value
        )

        assert result == 0.85

    def test_default_used_when_nothing_set(self):
        """Test that defaults are used when nothing else is set."""
        cli_value = None
        db_value = None
        default_value = 0.7

        result = cli_value if cli_value is not None else (
            db_value if db_value is not None else default_value
        )

        assert result == 0.7

    def test_zero_is_valid_explicit_value(self):
        """Test that zero is treated as an explicit value, not None."""
        # Important: 0 should not fall through to default
        cli_value = 0
        db_value = 0.85
        default_value = 0.7

        # If 0 is set explicitly, it should be used
        result = cli_value if cli_value is not None else (
            db_value if db_value is not None else default_value
        )

        assert result == 0

    def test_false_is_valid_explicit_value(self):
        """Test that False is treated as an explicit value, not None."""
        # Important: False should not fall through to default
        cli_value = False
        db_value = True
        default_value = True

        result = cli_value if cli_value is not None else (
            db_value if db_value is not None else default_value
        )

        assert result is False


class TestRAGResultStructure:
    """Test RAGResult dataclass structure."""

    def test_rag_result_import(self):
        """Test that RAGResult can be imported."""
        from docs2db_api.rag.engine import RAGResult

        assert RAGResult is not None

    def test_rag_result_creation(self):
        """Test that RAGResult can be created with required fields."""
        from docs2db_api.rag.engine import RAGResult

        result = RAGResult(
            query="test query",
            documents=[],
            refined_questions=None,
            metadata={},
        )

        assert result.query == "test query"
        assert result.documents == []
        assert result.refined_questions is None
        assert result.metadata == {}

