#!/usr/bin/env python3
"""
RAG CLI Interface
=================

Command-line interface for testing and using the Universal RAG Engine.
Useful for development, testing, and simple integrations.

"""

import argparse
import asyncio
import sys

import structlog

from docs2db_api.embeddings import EMBEDDING_CONFIGS
from docs2db_api.rag.engine import RAGConfig, UniversalRAGEngine

logger = structlog.get_logger(__name__)


async def search_command(args):
    """Handle search command"""
    config = RAGConfig(
        model_name=args.model,
        similarity_threshold=args.threshold,
        max_chunks=args.limit,
        enable_question_refinement=args.refine,
        enable_hybrid_search=args.hybrid,
    )

    engine = UniversalRAGEngine(config)

    try:
        logger.info("🔍 Searching", query=args.query, model=args.model, threshold=args.threshold, limit=args.limit)
        logger.info("=" * 60)

        result = await engine.search_documents(args.query)

        logger.info("✅ Found documents", count=len(result.documents))

        if result.refined_questions:
            logger.info("🎯 Refined Questions", questions=result.refined_questions)

        logger.info("📄 Documents found")
        for i, doc in enumerate(result.documents, 1):
            logger.info(
                "Document",
                index=i,
                similarity=doc['similarity_score'],
                source=doc['document_path'],
                text_preview=doc['text'][:300] + ('...' if len(doc['text']) > 300 else '')
            )

        if result.metadata:
            logger.info("📈 Metadata", metadata=result.metadata)

    except Exception as e:
        logger.error("❌ Search failed", error=str(e))
        return 1
    finally:
        await engine.close()

    return 0


async def test_command(args):
    """Handle test command"""
    logger.info("🧪 Running RAG Engine Tests")
    logger.info("=" * 60)

    test_queries = [
        "How do I configure SSH on RHEL?",
        "What are the system requirements for Red Hat Enterprise Linux?",
        "How to manage users and groups?",
        "Configure firewall rules",
        "Install packages with yum",
    ]

    config = RAGConfig(
        model_name=args.model,
        similarity_threshold=0.6,  # Lower threshold for testing
        max_chunks=3,
        enable_question_refinement=True,
    )

    engine = UniversalRAGEngine(config)

    try:
        for i, query in enumerate(test_queries, 1):
            logger.info("🔍 Test query", test_number=i, query=query)

            try:
                result = await engine.search_documents(query)
                logger.info("✅ Test result", documents_found=len(result.documents))

                if result.documents:
                    best_score = max(
                        doc["similarity_score"] for doc in result.documents
                    )
                    logger.info("📊 Best similarity", score=best_score)

            except Exception as e:
                logger.error("❌ Test failed", error=str(e))

        logger.info("✅ Test completed successfully")

    except Exception as e:
        logger.error("❌ Test failed", error=str(e))
        return 1
    finally:
        await engine.close()

    return 0


async def benchmark_command(args):
    """Handle benchmark command"""
    logger.info("⚡ Running RAG Engine Benchmark")
    logger.info("=" * 60)

    import time

    queries = [
        "SSH configuration",
        "User management",
        "Package installation",
        "System monitoring",
        "Network configuration",
    ] * args.iterations  # Repeat queries

    config = RAGConfig(
        model_name=args.model,
        max_chunks=5,
        enable_question_refinement=False,  # Disable for speed
    )

    engine = UniversalRAGEngine(config)

    try:
        start_time = time.time()
        total_docs = 0

        for i, query in enumerate(queries, 1):
            result = await engine.search_documents(query)
            total_docs += len(result.documents)

            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                logger.info("Benchmark progress", processed=i, total=len(queries), rate=rate)

        end_time = time.time()
        elapsed = end_time - start_time

        logger.info("📊 Benchmark Results", 
                   total_queries=len(queries),
                   total_time=elapsed,
                   avg_time_per_query=elapsed / len(queries),
                   queries_per_second=len(queries) / elapsed,
                   total_documents=total_docs,
                   avg_documents_per_query=total_docs / len(queries))

    except Exception as e:
        logger.error("❌ Benchmark failed", error=str(e))
        return 1
    finally:
        await engine.close()

    return 0


async def server_command(args):
    """Handle server command"""
    logger.info("🚀 Starting RAG API Server", port=args.port)

    try:
        import uvicorn

        from docs2db.rag.api import app

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except ImportError:
        logger.error("❌ FastAPI/uvicorn not available. Install with: pip install fastapi uvicorn")
        return 1
    except Exception as e:
        logger.error("❌ Server failed", error=str(e))
        return 1

    return 0


def models_command(args):
    """Handle models command"""
    logger.info("📋 Available Embedding Models")
    logger.info("=" * 60)

    for model_name, config in EMBEDDING_CONFIGS.items():
        logger.info("Model configuration", 
                   name=model_name,
                   model_id=config['model_id'],
                   dimensions=config['dimensions'],
                   provider=config['provider'],
                   keyword=config['keyword'])

    return 0


def main():
    parser = argparse.ArgumentParser(description="RAG CLI Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--model",
        default="granite-30m-english",
        choices=list(EMBEDDING_CONFIGS.keys()),
        help="Embedding model to use",
    )
    search_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Similarity threshold (0.0-1.0)"
    )
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results"
    )
    search_parser.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Disable question refinement",
    )
    search_parser.add_argument(
        "--no-hybrid", dest="hybrid", action="store_false", help="Disable hybrid search"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run test queries")
    test_parser.add_argument(
        "--model",
        default="granite-30m-english",
        choices=list(EMBEDDING_CONFIGS.keys()),
        help="Embedding model to use",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    benchmark_parser.add_argument(
        "--model",
        default="granite-30m-english",
        choices=list(EMBEDDING_CONFIGS.keys()),
        help="Embedding model to use",
    )
    benchmark_parser.add_argument(
        "--iterations", type=int, default=2, help="Number of iterations per query"
    )

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate handler
    if args.command == "search":
        return asyncio.run(search_command(args))
    elif args.command == "test":
        return asyncio.run(test_command(args))
    elif args.command == "benchmark":
        return asyncio.run(benchmark_command(args))
    elif args.command == "server":
        return asyncio.run(server_command(args))
    elif args.command == "models":
        return models_command(args)
    else:
        logger.error("Unknown command", command=args.command)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("❌ Unexpected error", error=str(e))
        sys.exit(1)
