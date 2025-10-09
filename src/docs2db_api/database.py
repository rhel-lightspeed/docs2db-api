"""Database operations for loading embeddings and chunks into PostgreSQL with pgvector."""

import asyncio
import json
import logging
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import psutil
import psycopg
import structlog
import yaml
from psycopg.sql import SQL, Identifier

from docs2db_api.exceptions import ConfigurationError, ContentError, DatabaseError

logger = structlog.get_logger()


def get_db_config() -> Dict[str, str]:
    """Parse postgres-compose.yml to get database connection parameters."""
    compose_file = Path(__file__).parent.parent.parent / "postgres-compose.yml"

    with open(compose_file, "r") as f:
        compose_data = yaml.safe_load(f)

    config = {"host": "localhost"}

    db_service = compose_data["services"]["db"]
    env = db_service["environment"]
    config["database"] = env["POSTGRES_DB"]
    config["user"] = env["POSTGRES_USER"]
    config["password"] = env["POSTGRES_PASSWORD"]

    # Extract port from ports mapping if available
    ports = db_service.get("ports", [])
    for port_mapping in ports:
        if isinstance(port_mapping, str) and ":5432" in port_mapping:
            host_port = port_mapping.split(":")[0]
            config["port"] = host_port
            break

    return config


class DatabaseManager:
    """Manages PostgreSQL database for pgvector storage."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    async def get_direct_connection(self):
        """Get a direct database connection."""
        connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return await psycopg.AsyncConnection.connect(connection_string)

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type from file extension."""
        suffix = file_path.suffix.lower()
        content_types = {
            ".json": "application/json",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".pdf": "application/pdf",
        }
        return content_types.get(suffix, "application/octet-stream")

    def _convert_timestamp(self, unix_timestamp: float):
        """Convert Unix timestamp to datetime object for PostgreSQL."""
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with await self.get_direct_connection() as conn:
            # Document stats
            doc_result = await conn.execute("SELECT COUNT(*) FROM documents")
            doc_row = await doc_result.fetchone()
            doc_count = doc_row[0] if doc_row else 0

            # Chunk stats
            chunk_result = await conn.execute("SELECT COUNT(*) FROM chunks")
            chunk_row = await chunk_result.fetchone()
            chunk_count = chunk_row[0] if chunk_row else 0

            # Embedding stats by model
            embedding_stats = await conn.execute(
                """
                SELECT model_name, COUNT(*) as count, AVG(dimensions) as avg_dimensions
                FROM embeddings
                GROUP BY model_name
                ORDER BY model_name
                """
            )
            embedding_models = {}
            async for row in embedding_stats:
                model_name, count, avg_dims = row
                embedding_models[model_name] = {
                    "count": count,
                    "dimensions": int(avg_dims) if avg_dims else 0,
                }

            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "embedding_models": embedding_models,
            }

    async def generate_manifest(self, output_file: str = "manifest.txt") -> bool:
        """Generate a manifest file with all unique source files in the database.

        Args:
            output_file: Path to the output manifest file

        Returns:
            bool: True if successful, False otherwise
        """
        async with await self.get_direct_connection() as conn:
            # Query for distinct document paths from documents table
            result = await conn.execute(
                """
                SELECT DISTINCT path
                FROM documents
                ORDER BY path
                """
            )

            # Write to manifest file iteratively
            manifest_path = Path(output_file)
            file_count = 0

            with open(manifest_path, "w") as f:
                async for row in result:
                    document_path = row[0]
                    f.write(f"{document_path}\n")
                    file_count += 1

            logger.info(
                f"Generated manifest with {file_count} unique document files",
                output_file=output_file,
            )
            return True

    async def search_similar(
        self,
        query_embedding: List[float],
        model_name: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        async with await self.get_direct_connection() as conn:
            results = await conn.execute(
                """
                SELECT
                    c.text,
                    c.metadata,
                    d.path,
                    d.filename,
                    e.embedding <=> %s::vector as distance,
                    1 - (e.embedding <=> %s::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                JOIN embeddings e ON c.id = e.chunk_id
                WHERE e.model_name = %s
                    AND 1 - (e.embedding <=> %s::vector) >= %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    model_name,
                    query_embedding,
                    similarity_threshold,
                    query_embedding,
                    limit,
                ),
            )

            similar_chunks = []
            async for row in results:
                text, metadata_json, doc_path, filename, distance, similarity = row

                # Handle metadata - it might be a dict already or a JSON string
                if metadata_json:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = metadata_json
                else:
                    metadata = {}

                similar_chunks.append({
                    "text": text,
                    "metadata": metadata,
                    "document_path": doc_path,
                    "document_filename": filename,
                    "distance": float(distance),
                    "similarity": float(similarity),
                })

            return similar_chunks


async def check_database_status(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Check database connectivity and display statistics."""
    db_defaults = get_db_config()
    host = host if host is not None else db_defaults["host"]
    port = port if port is not None else int(db_defaults["port"])
    db = db if db is not None else db_defaults["database"]
    user = user if user is not None else db_defaults["user"]
    password = password if password is not None else db_defaults["password"]

    logger.info(
        "\nCheck database status:\n"
        f"  Host    : {host}\n"
        f"  Port    : {port}\n"
        f"  Database: {db}\n"
        f"  user    : {user}"
    )

    # Suppress psycopg connection warnings for cleaner error messages
    logging.getLogger("psycopg.pool").setLevel(logging.ERROR)

    db_manager = DatabaseManager(
        host=host,
        port=port,
        database=db,
        user=user,
        password=password,
    )

    # Section 1: Test basic PostgreSQL server connectivity
    try:
        # First try a direct connection to catch auth errors immediately
        basic_connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/postgres"
        )

        async with await psycopg.AsyncConnection.connect(
            basic_connection_string, connect_timeout=5
        ) as conn:
            # Test basic connectivity
            result = await conn.execute("SELECT version(), now()")
            row = await result.fetchone()
            if row:
                _pg_version, _current_time = row
                logger.info("Database connection successful")

    except Exception as conn_error:
        # Handle server connectivity errors
        error_msg = str(conn_error).lower()
        if (
            "connection refused" in error_msg
            or "could not receive data" in error_msg
            or "couldn't get a connection" in error_msg
        ):
            logger.error("Database is not running. Start database with 'make db-up'")
        elif (
            "authentication failed" in error_msg
            or "no password supplied" in error_msg
            or "password authentication failed" in error_msg
            or "role" in error_msg
            and "does not exist" in error_msg
        ):
            logger.error("Database authentication failed. Check database credentials")
        else:
            logger.error("Database connection failed. Ensure PostgreSQL is running")

        raise DatabaseError(f"Database connection failed: {conn_error}") from conn_error

    # Section 2: Test target database connectivity
    try:
        # Now connect to our target database and test it
        async with await db_manager.get_direct_connection() as conn:
            # Test that we can actually query the target database
            await conn.execute("SELECT 1")
    except Exception as conn_error:
        # If we get here, PostgreSQL is running but our target database doesn't exist
        logger.error("Database does not exist. Create database or check name")
        raise DatabaseError("Database does not exist") from conn_error

    # If we get here, connection was successful, continue with checks

    # Check for pgvector extension
    async with await db_manager.get_direct_connection() as conn:
        ext_result = await conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        )
        ext_row = await ext_result.fetchone()
        if ext_row:
            _ext_name, ext_version = ext_row
            logger.info(f"pgvector extension found: version={ext_version}")
        else:
            logger.error(
                "pgvector extension not installed. "
                "Run 'uv run docs2db load' to initialize"
            )
            raise DatabaseError("pgvector extension not installed")

    # Check if tables exist
    async with await db_manager.get_direct_connection() as conn:
        tables_result = await conn.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('documents', 'chunks', 'embeddings')
                ORDER BY table_name
            """)
        tables = []
        async for row in tables_result:
            tables.append(row[0])

        if len(tables) == 3:
            logger.info("All required tables exist")
        elif len(tables) > 0:
            logger.error(
                "Partial schema found. Run 'uv run docs2db load' to initialize"
            )
            raise DatabaseError("Partial schema found")
        else:
            logger.error(
                "No docs2db tables found. Run 'uv run docs2db load' to initialize"
            )
            raise DatabaseError("No docs2db tables found")

    # Get database statistics
    stats = await db_manager.get_stats()

    total_embeddings = sum(
        model_info["count"] for model_info in stats["embedding_models"].values()
    )

    logger.info(
        "\nDatabase statistics summary:\n"
        f"  documents : {stats['documents']}\n"
        f"  chunks    : {stats['chunks']}\n"
        f"  embeddings: {total_embeddings}\n"
    )

    # Log embedding models breakdown
    if stats["embedding_models"]:
        for model_name, model_info in stats["embedding_models"].items():
            logger.info(
                "\nEmbedding model details:\n"
                f"model     : {model_name}\n"
                f"embeddings: {model_info['count']}\n"
                f"dimensions: {model_info['dimensions']}"
            )

    if stats["documents"] > 0:
        # Get recent activity
        async with await db_manager.get_direct_connection() as conn:
            recent_result = await conn.execute("""
                SELECT
                    filename,
                    created_at,
                    updated_at
                FROM documents
                ORDER BY updated_at DESC
                LIMIT 5
            """)

            file_str = ""
            async for row in recent_result:
                filename, created_at, updated_at = row
                file_str += f"  {filename}\n    created: {created_at.strftime('%Y-%m-%d %H:%M')}\n    updated: {updated_at.strftime('%Y-%m-%d %H:%M') if updated_at else 'Never'}\n"
            logger.info(f"\nRecent document activity (last 5)\n{file_str}")

        # Database size information
        async with await db_manager.get_direct_connection() as conn:
            size_result = await conn.execute(
                "SELECT pg_size_pretty(pg_database_size(%s)) as db_size", (db,)
            )
            size_row = await size_result.fetchone()
            if size_row:
                db_size = size_row[0]
                logger.info(f"Database size: {db_size}")

    logger.info("Database status check completed successfully")


async def _ensure_database_exists(
    host: str, port: int, db: str, user: str, password: str
) -> None:
    """Ensure the target database exists, create it if it doesn't."""

    # Connect to the default postgres database to check/create our target database
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/postgres"

    try:
        async with await psycopg.AsyncConnection.connect(
            connection_str,
            connect_timeout=5,
            autocommit=True,  # Needed for CREATE DATABASE
        ) as conn:
            # Check if our target database exists
            result = await conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (db,)
            )
            db_exists = await result.fetchone()

            if not db_exists:
                logger.info(f"Creating database '{db}'...")
                # Create the database (note: can't use parameters for database name in CREATE DATABASE)
                create_db_query = SQL("CREATE DATABASE {}").format(Identifier(db))
                await conn.execute(create_db_query)
                logger.info(f"Database '{db}' created successfully")

    except Exception as e:
        logger.error(f"Failed to ensure database exists: {e}")
        raise DatabaseError(f"Could not create database '{db}': {e}") from e


def restore_database(
    input_file: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Restore a PostgreSQL database from a dump file.

    Args:
        input_file: Input file path for the database dump
        host: Database host (auto-detected from compose file if None)
        port: Database port (auto-detected from compose file if None)
        db: Database name (auto-detected from compose file if None)
        user: Database user (auto-detected from compose file if None)
        password: Database password (auto-detected from compose file if None)
        verbose: Show psql output

    Returns:
        True if successful, False if errors occurred

    Raises:
        ConfigurationError: If psql is not found or configuration is invalid
        DatabaseError: If restore operation fails
    """
    config = get_db_config()
    host = host if host is not None else config["host"]
    port = port if port is not None else int(config["port"])
    db = db if db is not None else config["database"]
    user = user if user is not None else config["user"]
    password = password if password is not None else config["password"]

    input_path = Path(input_file)
    if not input_path.exists():
        raise DatabaseError(f"Dump file not found: {input_file}")

    logger.info(f"Restoring database dump: {user}@{host}:{port}/{db}")
    logger.info(f"Input file: {input_file}")

    # Build psql command
    cmd = [
        "psql",
        f"--host={host}",
        f"--port={port}",
        f"--username={user}",
        f"--dbname={db}",
        "--no-password",  # Use PGPASSWORD env var instead
        "--file",
        str(input_path),
    ]

    if not verbose:
        cmd.append("--quiet")

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    try:
        logger.info("Restoring database from dump...")

        # Run psql
        subprocess.run(
            cmd,
            env=env,
            capture_output=not verbose,
            text=True,
            check=True,
        )

        logger.info(f"Database restored successfully from: {input_file}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"psql failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        raise DatabaseError(
            f"Database restore failed with exit code {e.returncode}"
        ) from e
    except FileNotFoundError as e:
        raise ConfigurationError(
            "psql command not found. Please install PostgreSQL client tools."
        ) from e


async def generate_manifest(
    output_file: str = "manifest.txt",
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> bool:
    """Generate a manifest file with all unique source files in the database.

    Args:
        output_file: Path to the output manifest file
        host: Database host (auto-detected if not provided)
        port: Database port (auto-detected if not provided)
        db: Database name (auto-detected if not provided)
        user: Database user (auto-detected if not provided)
        password: Database password (auto-detected if not provided)

    Returns:
        bool: True if successful, False otherwise
    """
    config = get_db_config()
    host = host if host is not None else config["host"]
    port = port if port is not None else int(config["port"])
    db = db if db is not None else config["database"]
    user = user if user is not None else config["user"]
    password = password if password is not None else config["password"]

    db_manager = DatabaseManager(
        host=host,
        port=port,
        database=db,
        user=user,
        password=password,
    )

    return await db_manager.generate_manifest(output_file)
