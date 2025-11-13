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

from docs2db_api.embeddings import EMBEDDING_CONFIGS
from docs2db_api.exceptions import ConfigurationError, ContentError, DatabaseError

logger = structlog.get_logger()


def get_db_config() -> Dict[str, str]:
    """Get database connection parameters from multiple sources.

    Configuration precedence (highest to lowest):
    1. Environment variables (POSTGRES_HOST, POSTGRES_PORT, etc.)
    2. DATABASE_URL environment variable
    3. postgres-compose.yml in current working directory
    4. Default values (localhost:5432, user=postgres, db=ragdb)

    Raises:
        ConfigurationError: If both DATABASE_URL and individual POSTGRES_* vars are set

    Returns:
        Dict with keys: host, port, database, user, password
    """
    # Check for conflicting configuration sources
    has_database_url = bool(os.getenv("DATABASE_URL"))
    has_postgres_vars = any([
        os.getenv("POSTGRES_HOST"),
        os.getenv("POSTGRES_PORT"),
        os.getenv("POSTGRES_DB"),
        os.getenv("POSTGRES_USER"),
        os.getenv("POSTGRES_PASSWORD"),
    ])

    if has_database_url and has_postgres_vars:
        raise ConfigurationError(
            "Conflicting database configuration: both DATABASE_URL and individual "
            "POSTGRES_* environment variables are set. Please use one or the other."
        )

    # Start with sensible defaults
    config = {
        "host": "localhost",
        "port": "5432",
        "database": "ragdb",
        "user": "postgres",
        "password": "postgres",
    }

    # Try postgres-compose.yml in current working directory
    compose_file = Path.cwd() / "postgres-compose.yml"
    if compose_file.exists():
        try:
            with open(compose_file, "r") as f:
                compose_data = yaml.safe_load(f)

            db_service = compose_data.get("services", {}).get("db", {})
            env = db_service.get("environment", {})

            if "POSTGRES_DB" in env:
                config["database"] = env["POSTGRES_DB"]
            if "POSTGRES_USER" in env:
                config["user"] = env["POSTGRES_USER"]
            if "POSTGRES_PASSWORD" in env:
                config["password"] = env["POSTGRES_PASSWORD"]

            # Extract port from ports mapping if available
            ports = db_service.get("ports", [])
            for port_mapping in ports:
                if isinstance(port_mapping, str) and ":5432" in port_mapping:
                    host_port = port_mapping.split(":")[0]
                    config["port"] = host_port
                    break
        except Exception as e:
            # If compose file exists but can't be parsed, warn but continue with defaults
            logger.warning(f"Could not parse postgres-compose.yml: {e}")

    # DATABASE_URL takes precedence over compose file but not over individual vars
    if has_database_url:
        database_url = os.getenv("DATABASE_URL", "")
        try:
            # Parse postgresql://user:password@host:port/database
            # Support both postgresql:// and postgres:// schemes
            if database_url.startswith(("postgresql://", "postgres://")):
                # Remove scheme
                url_without_scheme = database_url.split("://", 1)[1]

                # Split into credentials@location and database
                if "@" in url_without_scheme:
                    credentials, location = url_without_scheme.split("@", 1)

                    # Parse credentials
                    if ":" in credentials:
                        config["user"], config["password"] = credentials.split(":", 1)
                    else:
                        config["user"] = credentials

                    # Parse location and database
                    if "/" in location:
                        host_port, config["database"] = location.split("/", 1)
                    else:
                        host_port = location

                    # Parse host and port
                    if ":" in host_port:
                        config["host"], config["port"] = host_port.split(":", 1)
                    else:
                        config["host"] = host_port
                else:
                    raise ConfigurationError(
                        f"Invalid DATABASE_URL format (missing @): {database_url}"
                    )
            else:
                raise ConfigurationError(
                    f"Invalid DATABASE_URL scheme. Expected postgresql:// or postgres://, "
                    f"got: {database_url.split('://')[0] if '://' in database_url else database_url}"
                )
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse DATABASE_URL: {e}. "
                f"Expected format: postgresql://user:password@host:port/database"
            ) from e

    # Individual environment variables override everything (highest precedence)
    if os.getenv("POSTGRES_HOST"):
        config["host"] = os.getenv("POSTGRES_HOST", "")
    if os.getenv("POSTGRES_PORT"):
        config["port"] = os.getenv("POSTGRES_PORT", "")
    if os.getenv("POSTGRES_DB"):
        config["database"] = os.getenv("POSTGRES_DB", "")
    if os.getenv("POSTGRES_USER"):
        config["user"] = os.getenv("POSTGRES_USER", "")
    if os.getenv("POSTGRES_PASSWORD"):
        config["password"] = os.getenv("POSTGRES_PASSWORD", "")

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

    async def get_model_id(self, conn, model_name: str) -> Optional[int]:
        """Get model ID by name."""
        result = await conn.execute(
            "SELECT id FROM models WHERE name = %s",
            (model_name,),
        )
        row = await result.fetchone()
        return row[0] if row else None

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

            # Embedding stats by model (using normalized models table)
            embedding_stats = await conn.execute(
                """
                SELECT m.name, COUNT(*) as count, m.dimensions
                FROM embeddings e
                JOIN models m ON e.model = m.id
                GROUP BY m.name, m.dimensions
                ORDER BY m.name
                """
            )
            embedding_models = {}
            async for row in embedding_stats:
                model_name, count, dimensions = row
                embedding_models[model_name] = {
                    "count": count,
                    "dimensions": dimensions,
                }

            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "embedding_models": embedding_models,
            }

    async def get_rag_settings(self) -> Optional[Dict[str, Any]]:
        """Get RAG settings from the database.
        
        Returns:
            Dictionary with RAG settings, or None if no settings exist
        """
        async with await self.get_direct_connection() as conn:
            try:
                result = await conn.execute(
                    """
                    SELECT refinement_prompt, enable_refinement, enable_reranking,
                           similarity_threshold, max_chunks, max_tokens_in_context,
                           refinement_questions_count
                    FROM rag_settings WHERE id = 1
                    """
                )
                row = await result.fetchone()
                
                if row is None:
                    return None
                
                return {
                    "refinement_prompt": row[0],
                    "enable_refinement": row[1],
                    "enable_reranking": row[2],
                    "similarity_threshold": row[3],
                    "max_chunks": row[4],
                    "max_tokens_in_context": row[5],
                    "refinement_questions_count": row[6],
                }
            except Exception as e:
                logger.warning(f"Could not retrieve RAG settings: {e}")
                return None

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

    async def search_vector(
        self,
        query_embedding: List[float],
        model_name: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity (pure semantic search)."""
        async with await self.get_direct_connection() as conn:
            # Get model_id from model name (which is the full model identifier)
            model_id = await self.get_model_id(conn, model_name)
            if model_id is None:
                logger.warning(f"Model '{model_name}' not found in database")
                return []

            results = await conn.execute(
                """
                SELECT
                    c.text,
                    c.metadata,
                    d.path,
                    d.filename,
                    c.chunk_index,
                    e.embedding <=> %s::vector as distance,
                    1 - (e.embedding <=> %s::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                JOIN embeddings e ON c.id = e.chunk_id
                WHERE e.model = %s
                    AND 1 - (e.embedding <=> %s::vector) >= %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    model_id,
                    query_embedding,
                    similarity_threshold,
                    query_embedding,
                    limit,
                ),
            )

            similar_chunks = []
            async for row in results:
                text, metadata_json, doc_path, filename, chunk_index, distance, similarity = row

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
                    "chunk_index": chunk_index,
                    "distance": float(distance),
                    "similarity": float(similarity),
                })

            return similar_chunks

    async def search_bm25(
        self,
        query_text: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for chunks using BM25 full-text search (pure lexical search)."""
        async with await self.get_direct_connection() as conn:
            results = await conn.execute(
                """
                SELECT
                    c.text,
                    c.metadata,
                    d.path,
                    d.filename,
                    c.chunk_index,
                    ts_rank(c.text_search_vector, websearch_to_tsquery('english', %s)) as rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.text_search_vector @@ websearch_to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query_text, query_text, limit),
            )

            bm25_chunks = []
            async for row in results:
                text, metadata_json, doc_path, filename, chunk_index, rank = row

                # Handle metadata
                if metadata_json:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = metadata_json
                else:
                    metadata = {}

                bm25_chunks.append({
                    "text": text,
                    "metadata": metadata,
                    "document_path": doc_path,
                    "document_filename": filename,
                    "chunk_index": chunk_index,
                    "bm25_rank": float(rank),
                })

            return bm25_chunks

    async def search_hybrid(
        self,
        query_embedding: List[float],
        query_text: str,
        model_name: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Search using hybrid approach: vector similarity + BM25 with Reciprocal Rank Fusion.
        
        This is the primary search method - combining semantic and lexical search.
        """
        async with await self.get_direct_connection() as conn:
            # Get model_id from model name (which is the full model identifier)
            model_id = await self.get_model_id(conn, model_name)
            if model_id is None:
                logger.warning(f"Model '{model_name}' not found in database")
                return []

            # Hybrid search with RRF combining vector and BM25 results
            results = await conn.execute(
                """
                WITH vector_results AS (
                    SELECT
                        c.id,
                        c.text,
                        c.metadata,
                        d.path,
                        d.filename,
                        c.chunk_index,
                        1 - (e.embedding <=> %s::vector) as similarity,
                        ROW_NUMBER() OVER (ORDER BY e.embedding <=> %s::vector) as vector_rank
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    JOIN embeddings e ON c.id = e.chunk_id
                    WHERE e.model = %s
                        AND 1 - (e.embedding <=> %s::vector) >= %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                ),
                bm25_results AS (
                    SELECT
                        c.id,
                        c.text,
                        c.metadata,
                        d.path,
                        d.filename,
                        c.chunk_index,
                        ts_rank(c.text_search_vector, websearch_to_tsquery('english', %s)) as bm25_rank,
                        ROW_NUMBER() OVER (ORDER BY ts_rank(c.text_search_vector, websearch_to_tsquery('english', %s)) DESC) as bm25_rank_position
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.text_search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY bm25_rank DESC
                    LIMIT %s
                ),
                combined AS (
                    SELECT
                        COALESCE(v.id, b.id) as id,
                        COALESCE(v.text, b.text) as text,
                        COALESCE(v.metadata, b.metadata) as metadata,
                        COALESCE(v.path, b.path) as path,
                        COALESCE(v.filename, b.filename) as filename,
                        COALESCE(v.chunk_index, b.chunk_index) as chunk_index,
                        v.similarity,
                        b.bm25_rank,
                        v.vector_rank,
                        b.bm25_rank_position,
                        -- Reciprocal Rank Fusion score
                        COALESCE(1.0 / (%s + v.vector_rank), 0.0) + COALESCE(1.0 / (%s + b.bm25_rank_position), 0.0) as rrf_score
                    FROM vector_results v
                    FULL OUTER JOIN bm25_results b ON v.id = b.id
                )
                SELECT
                    text,
                    metadata,
                    path,
                    filename,
                    chunk_index,
                    similarity,
                    bm25_rank,
                    rrf_score
                FROM combined
                ORDER BY rrf_score DESC
                LIMIT %s
                """,
                (
                    # vector_results params
                    query_embedding,
                    query_embedding,
                    model_id,
                    query_embedding,
                    similarity_threshold,
                    query_embedding,
                    limit * 2,  # Get more candidates for RRF
                    # bm25_results params
                    query_text,
                    query_text,
                    query_text,
                    limit * 2,  # Get more candidates for RRF
                    # RRF k constant
                    rrf_k,
                    rrf_k,
                    # Final limit
                    limit,
                ),
            )

            hybrid_chunks = []
            async for row in results:
                text, metadata_json, doc_path, filename, chunk_index, similarity, bm25_rank, rrf_score = row

                # Handle metadata
                if metadata_json:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = metadata_json
                else:
                    metadata = {}

                hybrid_chunks.append({
                    "text": text,
                    "metadata": metadata,
                    "document_path": doc_path,
                    "document_filename": filename,
                    "chunk_index": chunk_index,
                    "similarity": float(similarity) if similarity is not None else None,
                    "bm25_rank": float(bm25_rank) if bm25_rank is not None else None,
                    "rrf_score": float(rrf_score),
                })

            return hybrid_chunks

    def format_schema_change_display(self, change_data: dict) -> str:
        """Format a schema change record for display.

        Only includes fields that have meaningful values.
        """
        lines = []

        # Header with ID
        lines.append(f"\nUpdate #{change_data['id']}:")

        # Timestamp (always show)
        timestamp = (
            change_data["changed_at"].strftime("%Y-%m-%d %H:%M")
            if change_data["changed_at"]
            else "Unknown"
        )
        lines.append(f"  Timestamp      : {timestamp}")

        # User (only if set)
        if change_data["changed_by_user"]:
            lines.append(f"  User           : {change_data['changed_by_user']}")

        # Version (only if set)
        if change_data["changed_by_version"]:
            lines.append(f"  Version        : {change_data['changed_by_version']}")

        # Tool (only if set)
        if change_data["changed_by_tool"]:
            lines.append(f"  Tool           : {change_data['changed_by_tool']}")

        # Documents (only if added or deleted)
        if change_data["documents_added"] > 0:
            lines.append(f"  Documents added: {change_data['documents_added']}")
        if change_data["documents_deleted"] > 0:
            lines.append(f"  Documents deleted: {change_data['documents_deleted']}")

        # Chunks (only if added or deleted)
        if change_data["chunks_added"] > 0:
            lines.append(f"  Chunks added   : {change_data['chunks_added']}")
        if change_data["chunks_deleted"] > 0:
            lines.append(f"  Chunks deleted : {change_data['chunks_deleted']}")

        # Embeddings (only if added or deleted)
        if change_data["embeddings_added"] > 0:
            lines.append(f"  Embeds added   : {change_data['embeddings_added']}")
        if change_data["embeddings_deleted"] > 0:
            lines.append(f"  Embeds deleted : {change_data['embeddings_deleted']}")

        # Models added (only if any)
        if change_data["embedding_models_added"]:
            models_str = ", ".join(change_data["embedding_models_added"])
            lines.append(f"  Models added   : {models_str}")

        # Notes (only if set)
        if change_data["notes"]:
            lines.append(f"  Notes          : {change_data['notes']}")

        return "\n".join(lines)


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
                f"  model     : {model_name}\n"
                f"  dimensions: {model_info['dimensions']}\n"
                f"  embeddings: {model_info['count']}"
            )

    # Display schema metadata if available
    async with await db_manager.get_direct_connection() as conn:
        try:
            metadata_result = await conn.execute(
                "SELECT * FROM schema_metadata WHERE id = 1"
            )
            metadata_row = await metadata_result.fetchone()
            if metadata_row and metadata_result.description:
                columns = [desc[0] for desc in metadata_result.description]
                metadata = dict(zip(columns, metadata_row))

                logger.info(
                    "\nSchema Metadata:\n"
                    f"  Version        : {metadata['schema_version']}\n"
                    f"  Title          : {metadata['title'] or '(not set)'}\n"
                    f"  Description    : {metadata['description'] or '(not set)'}\n"
                    f"  Models         : {metadata['embedding_models_count']}\n"
                    f"  Last modified  : {metadata['last_modified_at'].strftime('%Y-%m-%d %H:%M') if metadata['last_modified_at'] else 'Unknown'}"
                )
        except Exception:
            # Schema metadata table doesn't exist yet
            pass

    # Display recent schema changes (last 5)
    async with await db_manager.get_direct_connection() as conn:
        try:
            changes_result = await conn.execute("""
                SELECT
                    id,
                    changed_at,
                    changed_by_tool,
                    changed_by_version,
                    changed_by_user,
                    documents_added,
                    documents_deleted,
                    chunks_added,
                    chunks_deleted,
                    embeddings_added,
                    embeddings_deleted,
                    embedding_models_added,
                    notes
                FROM schema_changes
                ORDER BY id DESC
                LIMIT 5
            """)

            changes = []
            async for row in changes_result:
                if changes_result.description:
                    columns = [desc[0] for desc in changes_result.description]
                    change_data = dict(zip(columns, row))
                    changes.append(change_data)

            if changes:
                logger.info("\nRecent Changes (last 5):")
                for change_data in changes:
                    logger.info(db_manager.format_schema_change_display(change_data))
        except Exception:
            # Schema changes table doesn't exist yet
            pass

    if stats["documents"] > 0:
        # Get recent activity
        async with await db_manager.get_direct_connection() as conn:
            recent_result = await conn.execute("""
                SELECT
                    path,
                    created_at,
                    updated_at
                FROM documents
                ORDER BY updated_at DESC
                LIMIT 5
            """)

            file_str = ""
            async for row in recent_result:
                path, created_at, updated_at = row
                # Strip /source.json suffix for cleaner display
                display_path = path.removesuffix("/source.json")
                file_str += f"  {display_path}\n    created: {created_at.strftime('%Y-%m-%d %H:%M')}\n    updated: {updated_at.strftime('%Y-%m-%d %H:%M') if updated_at else 'Never'}\n"
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

    # Display RAG settings if configured
    rag_settings = await db_manager.get_rag_settings()
    if rag_settings:
        # Format non-None settings for display
        settings_lines = []
        if rag_settings["enable_refinement"] is not None:
            settings_lines.append(f"  enable_refinement         : {rag_settings['enable_refinement']}")
        if rag_settings["enable_reranking"] is not None:
            settings_lines.append(f"  enable_reranking          : {rag_settings['enable_reranking']}")
        if rag_settings["similarity_threshold"] is not None:
            settings_lines.append(f"  similarity_threshold      : {rag_settings['similarity_threshold']}")
        if rag_settings["max_chunks"] is not None:
            settings_lines.append(f"  max_chunks                : {rag_settings['max_chunks']}")
        if rag_settings["max_tokens_in_context"] is not None:
            settings_lines.append(f"  max_tokens_in_context     : {rag_settings['max_tokens_in_context']}")
        if rag_settings["refinement_questions_count"] is not None:
            settings_lines.append(f"  refinement_questions_count: {rag_settings['refinement_questions_count']}")
        if rag_settings["refinement_prompt"] is not None:
            # Truncate prompt if too long
            prompt_preview = rag_settings["refinement_prompt"][:100]
            if len(rag_settings["refinement_prompt"]) > 100:
                prompt_preview += "..."
            settings_lines.append(f"  refinement_prompt         : {prompt_preview}")
        
        if settings_lines:
            logger.info("\nRAG settings:\n" + "\n".join(settings_lines))

    logger.info("Database status check complete")


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
