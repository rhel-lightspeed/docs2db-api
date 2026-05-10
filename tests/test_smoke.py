"""Basic import and startup smoke tests — no external services required."""

import pytest


class TestModuleImports:
    def test_import_config(self):
        import docs2db_api.config

        assert docs2db_api.config is not None

    def test_import_database(self):
        import docs2db_api.database

        assert docs2db_api.database is not None

    def test_import_exceptions(self):
        import docs2db_api.exceptions

        assert docs2db_api.exceptions is not None

    def test_import_embeddings(self):
        import docs2db_api.embeddings

        assert docs2db_api.embeddings is not None

    def test_import_reranker(self):
        import docs2db_api.reranker

        assert docs2db_api.reranker is not None


class TestConfigDefaults:
    def test_settings_instance_exists(self):
        from docs2db_api.config import settings

        assert settings is not None

    def test_database_defaults(self):
        from docs2db_api.config import settings

        assert settings.database.host == "localhost"
        assert settings.database.port == 5432
        assert settings.database.database == "ragdb"
        assert settings.database.user == "postgres"

    def test_rag_defaults(self):
        from docs2db_api.config import settings

        assert settings.rag.similarity_threshold == 0.7
        assert settings.rag.max_chunks == 10
        assert settings.rag.max_tokens_in_context == 4096
        assert settings.rag.enable_question_refinement is True
        assert settings.rag.enable_reranking is True

    def test_llm_defaults(self):
        from docs2db_api.config import settings

        assert settings.llm.model == "qwen2.5:7b-instruct"
        assert settings.llm.timeout == pytest.approx(30.0)

    def test_database_url_defaults_to_none(self):
        from docs2db_api.config import settings

        assert settings.database.url is None


class TestCLIApp:
    def test_typer_app_exists(self):
        from docs2db_api.docs2db_api import app

        assert app is not None

    def test_database_manager_class_exists(self):
        from docs2db_api.database import DatabaseManager

        assert DatabaseManager is not None

    def test_get_db_config_callable(self):
        from docs2db_api.database import get_db_config

        assert callable(get_db_config)

    def test_exception_hierarchy(self):
        from docs2db_api.exceptions import ConfigurationError, DatabaseError, Docs2DBException

        assert issubclass(DatabaseError, Docs2DBException)
        assert issubclass(ConfigurationError, Docs2DBException)
