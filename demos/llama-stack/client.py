#!/usr/bin/env python3
"""
Docs2DB RAG Demo Client - Simplified
===================================

Simple client that calls the RAG tool and displays search results.

python client.py --query "What is Solar Power?"

"""

import argparse
import sys

from llama_stack_client import LlamaStackClient


class Docs2DBRAGDemoClient:
    def __init__(self, base_url="http://localhost:8321"):
        self.base_url = base_url
        self.client = LlamaStackClient(base_url=base_url)

    def test_connection(self):
        """Test connection to Llama Stack server"""
        try:
            models = self.client.models.list()
            return True
        except Exception:
            return False

    def call_rag_tool(self, query, **kwargs):
        """Call the RAG tool using Llama Stack client"""
        print(f"🔍 Calling RAG tool with query: '{query}'")

        # Default parameters
        params = {
            "query": query,
            "similarity_threshold": kwargs.get("similarity_threshold", 0.7),
            "enable_hybrid_search": kwargs.get("enable_hybrid_search", True),
            "enable_question_refinement": kwargs.get("enable_question_refinement", True),
            "max_chunks": kwargs.get("max_chunks", 5),
        }

        print(f"📊 Parameters:")
        for key, value in params.items():
            print(f"   • {key}: {value}")
        print()

        try:
            print("   Calling search_documents tool...")
            result = self.client.tool_runtime.invoke_tool(
                tool_name="search_documents",
                kwargs=params,
            )
            return result
        except Exception as e:
            print(f"   search_documents failed: {e}")
            return None

    def display_rag_results(self, result):
        """Display RAG search results in a clean format"""
        if not result:
            print("❌ No results to display")
            return

        print("📋 RAG Search Results:")
        print("=" * 60)

        # Extract the actual data from the result
        result_data = None
        if hasattr(result, "content"):
            result_data = result.content
        elif hasattr(result, "data"):
            result_data = result.data
        else:
            result_data = result

        # Check if it's already a formatted string (from the tool)
        if isinstance(result_data, str) and "Found" in result_data and "relevant documents" in result_data:
            print("✅ RAG Search Results:")
            print(result_data)
            return

        # Try to parse as JSON for structured data
        try:
            import json
            if isinstance(result_data, str):
                data = json.loads(result_data)
            else:
                data = result_data

            # Display summary
            if "documents" in data:
                documents = data["documents"]
                print(f"✅ Found {len(documents)} documents")
                
                if "refined_questions" in data and data["refined_questions"]:
                    print(f"\n🎯 Refined Questions:")
                    refined_questions = data["refined_questions"]
                    if isinstance(refined_questions, str):
                        print(f"   {refined_questions}")
                    else:
                        for i, q in enumerate(refined_questions, 1):
                            print(f"   {i}. {q}")
                    print()

                # Display each document
                print(f"📄 Document Details:")
                for i, doc in enumerate(documents, 1):
                    print(f"\n{i}. Similarity: {doc.get('similarity_score', 'N/A'):.3f}")
                    print(f"   Source: {doc.get('document_path', 'N/A')}")
                    text = doc.get('text', '')
                    preview = text[:200] + ('...' if len(text) > 200 else '')
                    print(f"   Preview: {preview}")

                # Display metadata if available
                if "metadata" in data:
                    metadata = data["metadata"]
                    print(f"\n📈 Search Metadata:")
                    for key, value in metadata.items():
                        print(f"   {key}: {value}")

            else:
                print("❌ No documents found in result")
                print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

        except Exception as e:
            print(f"❌ Error parsing results: {e}")
            print(f"Raw result: {result_data}")

    def run_query(self, query, **kwargs):
        """Run a complete RAG query and display results"""
        print("🔍 Docs2DB RAG Search")
        print("=" * 50)
        print(f"Query: {query}")
        print()

        # Test connection first
        if not self.test_connection():
            print("❌ Cannot connect to Llama Stack server")
            print("Make sure the server is running on", self.base_url)
            return False

        print("✅ Connected to Llama Stack server")
        print()

        # Call RAG tool
        result = self.call_rag_tool(query, **kwargs)
        
        if result:
            self.display_rag_results(result)
            return True
        else:
            print("❌ RAG search failed")
            return False


def main():
    parser = argparse.ArgumentParser(description="Docs2DB RAG Search Demo")
    parser.add_argument(
        "--query",
        default="How do I configure SSH key-based authentication?",
        help="Query to search for (default: SSH configuration question)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for search (default: 0.7)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5,
        help="Maximum number of chunks to return (default: 5)",
    )
    parser.add_argument(
        "--disable-hybrid",
        action="store_true",
        help="Disable hybrid search",
    )
    parser.add_argument(
        "--disable-refinement",
        action="store_true",
        help="Disable question refinement",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8321",
        help="Llama Stack server URL (default: http://localhost:8321)",
    )

    args = parser.parse_args()

    # Create client and run query
    client = Docs2DBRAGDemoClient(base_url=args.server_url)
    
    kwargs = {
        "similarity_threshold": args.similarity_threshold,
        "max_chunks": args.max_chunks,
        "enable_hybrid_search": not args.disable_hybrid,
        "enable_question_refinement": not args.disable_refinement,
    }

    success = client.run_query(args.query, **kwargs)
    
    if success:
        print("\n✅ RAG search completed successfully!")
    else:
        print("\n❌ RAG search failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()