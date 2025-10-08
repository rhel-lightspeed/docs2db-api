#!/usr/bin/env python3
"""
RAG CLI Interface
=================

Command-line interface for testing and using the Universal RAG Engine.
Useful for development, testing, and simple integrations.

Usage:
    python -m codex.rag.cli search "How do I configure SSH?"
    python -m codex.rag.cli test
    python -m codex.rag.cli server --port 8000
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

from codex.embeddings import EMBEDDING_CONFIGS
from codex.rag.engine import RAGConfig, UniversalRAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        print(f"🔍 Searching: {args.query}")
        print(
            f"📊 Model: {args.model}, Threshold: {args.threshold}, Limit: {args.limit}"
        )
        print("=" * 60)

        result = await engine.search_documents(args.query)

        print(f"✅ Found {len(result.documents)} documents")

        if result.refined_questions:
            print(f"\n🎯 Refined Questions:\n{result.refined_questions}")

        print(f"\n📄 Documents:")
        for i, doc in enumerate(result.documents, 1):
            print(f"\n{i}. Similarity: {doc['similarity_score']:.3f}")
            print(f"   Source: {doc['document_path']}")
            print(
                f"   Text: {doc['text'][:300]}{'...' if len(doc['text']) > 300 else ''}"
            )

        if result.metadata:
            print(f"\n📈 Metadata:")
            for key, value in result.metadata.items():
                if key != "documents_details":  # Skip verbose details
                    print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return 1
    finally:
        await engine.close()

    return 0


async def test_command(args):
    """Handle test command"""
    print("🧪 Running RAG Engine Tests")
    print("=" * 60)

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
            print(f"\n🔍 Test {i}: {query}")

            try:
                result = await engine.search_documents(query)
                print(f"   ✅ Found {len(result.documents)} documents")

                if result.documents:
                    best_score = max(
                        doc["similarity_score"] for doc in result.documents
                    )
                    print(f"   📊 Best similarity: {best_score:.3f}")

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        print(f"\n✅ Test completed successfully")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    finally:
        await engine.close()

    return 0


async def benchmark_command(args):
    """Handle benchmark command"""
    print("⚡ Running RAG Engine Benchmark")
    print("=" * 60)

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
                print(
                    f"   Processed {i}/{len(queries)} queries ({rate:.1f} queries/sec)"
                )

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\n📊 Benchmark Results:")
        print(f"   Total queries: {len(queries)}")
        print(f"   Total time: {elapsed:.2f} seconds")
        print(f"   Average time per query: {elapsed / len(queries):.3f} seconds")
        print(f"   Queries per second: {len(queries) / elapsed:.1f}")
        print(f"   Total documents retrieved: {total_docs}")
        print(f"   Average documents per query: {total_docs / len(queries):.1f}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1
    finally:
        await engine.close()

    return 0


async def server_command(args):
    """Handle server command"""
    print(f"🚀 Starting RAG API Server on port {args.port}")

    try:
        import uvicorn

        from codex.rag.api import app

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except ImportError:
        print(
            "❌ FastAPI/uvicorn not available. Install with: pip install fastapi uvicorn"
        )
        return 1
    except Exception as e:
        print(f"❌ Server failed: {e}")
        return 1

    return 0


def models_command(args):
    """Handle models command"""
    print("📋 Available Embedding Models")
    print("=" * 60)

    for model_name, config in EMBEDDING_CONFIGS.items():
        print(f"\n{model_name}:")
        print(f"   Model ID: {config['model_id']}")
        print(f"   Dimensions: {config['dimensions']}")
        print(f"   Provider: {config['provider']}")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Keyword: {config['keyword']}")

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
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
