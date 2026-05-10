"""Integration smoke tests for PostgreSQL connectivity and pgvector availability."""

import psycopg
import pytest

from docs2db_api.database import DatabaseManager


@pytest.mark.integration
class TestDatabaseConnection:
    def test_sync_connection(self, skip_if_no_pg, test_db_config):
        cfg = test_db_config
        conn_string = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        with psycopg.Connection.connect(conn_string, connect_timeout=5) as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result is not None
            assert result[0] == 1

    async def test_async_connection(self, pg_connection):
        result = await pg_connection.execute("SELECT 1")
        row = await result.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_pgvector_extension_is_available(self, pg_connection):
        result = await pg_connection.execute("SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector'")
        row = await result.fetchone()
        assert row is not None
        assert row[0] >= 1

    async def test_database_manager_connection(self, pg_connection, test_db_config):
        cfg = test_db_config
        manager = DatabaseManager(
            host=cfg["host"],
            port=cfg["port"],
            database=cfg["database"],
            user=cfg["user"],
            password=cfg["password"],
        )
        async with await manager.get_direct_connection() as conn:
            result = await conn.execute("SELECT version()")
            row = await result.fetchone()
            assert row is not None
            assert "PostgreSQL" in row[0]
