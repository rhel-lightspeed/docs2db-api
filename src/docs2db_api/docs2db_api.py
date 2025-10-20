"""RAG Pipeline Tools for docs2db"""

import asyncio
from typing import Annotated, Optional

import structlog
import typer

from docs2db_api.database import (
    check_database_status,
    generate_manifest,
    restore_database,
)
from docs2db_api.exceptions import Docs2DBException
from docs2db_api.rag.engine import RAGConfig, UniversalRAGEngine

logger = structlog.get_logger(__name__)

app = typer.Typer(help="Make a RAG Database from source content")


@app.command(name="db-status")
def db_status(
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
) -> None:
    """Check database status and display statistics."""
    try:
        asyncio.run(
            check_database_status(
                host=host,
                port=port,
                db=db,
                user=user,
                password=password,
            )
        )
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command(name="db-restore")
def db_restore(
    input_file: Annotated[
        str, typer.Argument(help="Input file path for the database dump")
    ],
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Show psql output")] = False,
) -> None:
    """Restore a PostgreSQL database from a dump file."""
    try:
        if not restore_database(
            input_file=input_file,
            host=host,
            port=port,
            db=db,
            user=user,
            password=password,
            verbose=verbose,
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def manifest(
    output_file: Annotated[
        str, typer.Option(help="Output file path for the manifest")
    ] = "manifest.txt",
    host: Annotated[
        Optional[str],
        typer.Option(help="Database host (auto-detected from compose file)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="Database port (auto-detected from compose file)"),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(help="Database name (auto-detected from compose file)"),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(help="Database user (auto-detected from compose file)"),
    ] = None,
    password: Annotated[
        Optional[str],
        typer.Option(help="Database password (auto-detected from compose file)"),
    ] = None,
) -> None:
    """Generate a manifest file with all unique source files from the database."""
    try:
        if not asyncio.run(
            generate_manifest(
                output_file=output_file,
                host=host,
                port=port,
                db=db,
                user=user,
                password=password,
            )
        ):
            raise typer.Exit(1)
    except Docs2DBException as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def query(
    query_text: Annotated[str, typer.Argument(help="Search query")],
    model: Annotated[
        str,
        typer.Option(help="Embedding model to use"),
    ] = "granite-30m-english",
    limit: Annotated[int, typer.Option(help="Maximum number of results")] = 10,
    threshold: Annotated[
        float, typer.Option(help="Similarity threshold (0.0-1.0)")
    ] = 0.7,
    refine: Annotated[
        bool, typer.Option(help="Enable question refinement")
    ] = True,
) -> None:
    """Search documents using RAG engine with hybrid search and reranking."""
    try:
        config = RAGConfig(
            model_name=model,
            similarity_threshold=threshold,
            max_chunks=limit,
            enable_question_refinement=refine,
        )

        async def run_query():
            engine = UniversalRAGEngine(config)
            try:
                logger.info("🔍 Searching", query=query_text, model=model, threshold=threshold, limit=limit)
                logger.info("=" * 60)

                result = await engine.search_documents(query_text)

                logger.info("✅ Found documents", count=len(result.documents))

                if result.metadata:
                    metadata_lines = ["📈 Metadata:"]
                    for key, value in result.metadata.items():
                        metadata_lines.append(f"{key:<20} {value}")
                    logger.info("\n".join(metadata_lines))

                if result.refined_questions:
                    logger.info("🎯 Refined Questions", questions=result.refined_questions)

                logger.info("📄 Documents found")
                for i, doc in enumerate(result.documents, 1):
                    text_preview = doc['text'][:300] + ('...' if len(doc['text']) > 300 else '')
                    logger.info(
                        f"Document\n{text_preview}",
                        index=i,
                        similarity=doc['similarity_score'],
                        source=doc['document_path']
                    )

            finally:
                await engine.close()

        asyncio.run(run_query())

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise typer.Exit(1)

