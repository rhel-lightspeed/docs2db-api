#!/usr/bin/env python3
"""
Docs2DB RAG Demo Client
====================

The client uses a manual workflow:
1. Call RAG tool directly to get search results
2. Display the RAG features in action
3. Use RAG results in an inference call to get enhanced responses

Usage:
    python client.py [--query "your question"]

Example:
    python client.py --query "How do I configure SSH on RHEL?"
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
            # Try to list models using Llama Stack client
            models = self.client.models.list()
            return True
        except Exception:
            return False

    def check_available_tools(self):
        """Check what tools and tool groups are actually available"""
        print("Checking available tools and tool groups...")

        try:
            # Check tool groups using Llama Stack client
            toolgroups = self.client.toolgroups.list()
            print("Available tool groups:")
            if hasattr(toolgroups, "data") and toolgroups.data:
                for tg in toolgroups.data:
                    identifier = getattr(tg, "identifier", "unknown")
                    print(f"   • {identifier}")
            else:
                print(f"   Response: {toolgroups}")

            # Check tools using Llama Stack client
            tools = self.client.tools.list()
            print("Available tools:")
            if hasattr(tools, "data") and tools.data:
                for tool in tools.data:
                    tool_id = getattr(tool, "identifier", "unknown")
                    toolgroup_id = getattr(tool, "toolgroup_id", "none")
                    print(f"   • {tool_id} (group: {toolgroup_id})")
            else:
                print(f"   Response: {tools}")

        except Exception as e:
            print(f"Error checking tools: {e}")

        print()

    def call_rag_tool(self, query, **kwargs):
        """Call the RAG tool using Llama Stack client"""
        print(f"Calling RAG tool with query: '{query}'")

        # Default parameters that showcase RAG features
        params = {
            "query": query,
            "model_name": "granite-30m-english",
            "max_chunks": 5,
            "similarity_threshold": 0.7,  # Similarity Post Processing
            "enable_question_refinement": True,  # Query Refinement
            "enable_hybrid_search": True,  # Hybrid Search
            **kwargs,
        }

        print(f"Parameters:")
        print(f"   • similarity_threshold: {params['similarity_threshold']}")
        print(f"   • enable_hybrid_search: {params['enable_hybrid_search']}")
        print(
            f"   • enable_question_refinement: {params['enable_question_refinement']}"
        )
        print(f"   • max_chunks: {params['max_chunks']}")
        print()

        # Use search_documents for modular RAG (retrieval only)
        try:
            print("   Calling search_documents tool...")

            result = self.client.tool_runtime.invoke_tool(
                tool_name="search_documents", kwargs=params
            )
            return result

        except Exception as e:
            print(f"   search_documents failed: {e}")

        print("RAG tool invocation failed")
        return None

    def display_rag_features(self, result):
        """Display how the RAG features are working"""
        if not result:
            print("No results to display")
            return

        # Check if RAG returned no content
        if hasattr(result, "content") and result.content is None:
            print("ERROR: RAG search found no relevant documents")
            print("This could be due to:")
            print("- Similarity threshold too high")
            print("- No documents matching the query")
            print("- Database not properly loaded")
            return False

        print("RAG Features in Action:")
        print("=" * 50)

        # Convert ToolInvocationResult to dict if needed
        if hasattr(result, "content"):
            # Handle ToolInvocationResult object
            result_data = result.content
        else:
            result_data = result

        # Debug: print the actual result structure
        print(f"Debug - Result type: {type(result)}")
        print(f"Debug - Result data: {result_data}")
        print()

        # Feature 1: Query Refinement
        refined_questions = None

        # Check in the main result data
        if isinstance(result_data, dict):
            refined_questions = result_data.get("refined_questions")
        elif hasattr(result_data, "refined_questions"):
            refined_questions = getattr(result_data, "refined_questions", None)

        # Also check in the ToolInvocationResult metadata
        if not refined_questions and hasattr(result, "metadata") and result.metadata:
            refined_questions = result.metadata.get("refined_questions")

        if refined_questions:
            print("1. Query Refinement:")
            print("   Original query was expanded into multiple refined questions:")

            # Split the refined questions by newlines
            questions_list = refined_questions.strip().split("\n")
            for i, refined_q in enumerate(questions_list, 1):
                if refined_q.strip():  # Only print non-empty lines
                    print(f"   {i}. {refined_q.strip()}")
            print()
        else:
            print("1. Query Refinement: Not used for this query")
            print()

        # Feature 2: Hybrid Search
        print("2. Hybrid Search:")
        print("   Combined vector similarity + keyword search")
        search_metadata = None
        if isinstance(result_data, dict):
            search_metadata = result_data.get("search_metadata")
        elif hasattr(result_data, "search_metadata"):
            search_metadata = getattr(result_data, "search_metadata", None)

        if search_metadata:
            if isinstance(search_metadata, dict):
                vector_results = search_metadata.get("vector_results")
                keyword_results = search_metadata.get("keyword_results")
            else:
                vector_results = getattr(search_metadata, "vector_results", None)
                keyword_results = getattr(search_metadata, "keyword_results", None)

            if vector_results and keyword_results:
                print(f"   • Vector similarity results: {len(vector_results)}")
                print(f"   • Keyword search results: {len(keyword_results)}")
                print("   • Results merged and re-ranked")
            else:
                print("   • (detailed search breakdown not available)")
        else:
            print("   (no search metadata returned)")
        print()

        # Feature 3: Similarity Post Processing
        chunks = []
        # Get document details from metadata
        if hasattr(result, "metadata") and result.metadata:
            chunks = result.metadata.get("documents_details", [])
        elif isinstance(result_data, dict):
            chunks = result_data.get("chunks", [])
        elif hasattr(result_data, "chunks"):
            chunks = getattr(result_data, "chunks", [])

        if chunks:
            print("3. Similarity Post Processing:")
            print(f"  Filtered results by similarity threshold (≥0.7):")
            for i, chunk in enumerate(chunks, 1):
                if isinstance(chunk, dict):
                    similarity = chunk.get("similarity_score", "N/A")
                    # documents_details uses 'text_preview', not 'content'
                    content = chunk.get("text_preview", chunk.get("content", ""))
                else:
                    similarity = getattr(chunk, "similarity_score", "N/A")
                    content = getattr(
                        chunk, "text_preview", getattr(chunk, "content", "")
                    )
                # Content is already previewed in documents_details
                content_preview = (
                    content[:100] + "..." if len(content) > 100 else content
                )
                print(f"   {i}. Score: {similarity} - {content_preview}")
            print()
        else:
            print("3. Similarity Post Processing: No chunks met threshold")
            print()

    def call_inference_without_rag(self, query):
        """Call Llama Stack inference without RAG context to show baseline"""
        try:
            print("Generating baseline response WITHOUT RAG context...")

            response = self.client.inference.chat_completion(
                model_id="ollama/qwen2.5:7b-instruct",
                messages=[{"role": "user", "content": query}],
                sampling_params={"strategy": {"type": "greedy"}, "max_tokens": 512},
            )

            if hasattr(response, "completion_message"):
                content = response.completion_message.content
                print("Baseline response generated!")
                return content
            else:
                print("Unexpected response format")
                return None

        except Exception as e:
            print(f"Error calling inference without RAG: {e}")
            return None

    def call_inference_with_rag(self, original_query, rag_result):
        """Use RAG results in inference call using standard Llama Stack pattern"""
        if not rag_result:
            print("No RAG results available for inference")
            return None

        print("Generating enhanced response using standard Llama Stack pattern...")

        # Extract the RAG content from the ToolInvocationResult
        if hasattr(rag_result, "content"):
            rag_content = rag_result.content
        else:
            rag_content = str(rag_result)

        # Check if RAG content is None (no documents found)
        if rag_content is None:
            print("ERROR: Cannot generate enhanced response - RAG returned no content")
            return None

        try:
            # Use the standard Llama Stack pattern:
            # 1. User asks a question
            # 2. Tool response provides context
            # 3. Model generates answer based on both

            response = self.client.inference.chat_completion(
                model_id="ollama/qwen2.5:7b-instruct",
                messages=[
                    {"role": "user", "content": original_query},
                    {
                        "role": "tool",
                        "call_id": "search_documents_1",
                        "content": rag_content,
                    },
                ],
                sampling_params={"strategy": {"type": "greedy"}, "max_tokens": 512},
            )

            if hasattr(response, "completion_message"):
                content = response.completion_message.content
                print("Enhanced response generated using standard Llama Stack pattern!")
                return content
            else:
                print("Unexpected response format")
                return None

        except Exception as e:
            print(f"Error calling inference: {e}")
            return None

    def run_demo(self, query):
        """Run the complete RAG demonstration"""
        print("Docs2DB RAG Feature Demonstration")
        print("=" * 60)
        print(f"Query: {query}")
        print()

        # Step 1: Test connection
        print("Testing connection to Llama Stack server...")
        if not self.test_connection():
            print(f"Cannot connect to Llama Stack server at {self.base_url}")
            print(
                "   Make sure the server is running with: uv run python start_server.py"
            )
            return False
        print("Connected to Llama Stack server")
        print()

        # Step 2: Check available tools
        self.check_available_tools()

        # Step 3: Call RAG tool
        rag_result = self.call_rag_tool(query)
        if not rag_result:
            return False

        # Step 3: Display RAG features and check for valid results
        display_result = self.display_rag_features(rag_result)
        if display_result is False:
            print("\nDemo stopped: RAG returned no relevant documents")
            return False

        # Step 4: Compare responses - WITHOUT RAG vs WITH RAG
        print("\n" + "=" * 80)
        print("RAG EFFECTIVENESS COMPARISON")
        print("=" * 80)

        # First, get baseline response without RAG
        print("\nBASELINE: Response without RAG context")
        print("-" * 50)
        baseline_response = self.call_inference_without_rag(query)

        if baseline_response:
            print(f"Baseline Response:\n{baseline_response}")
        else:
            print("Failed to get baseline response")

        print("\n" + "-" * 80)

        # Then, get enhanced response with RAG
        print("\nENHANCED: Response with RAG context")
        print("-" * 50)
        enhanced_response = self.call_inference_with_rag(query, rag_result)

        if enhanced_response:
            print(f"Enhanced Response:\n{enhanced_response}")
        else:
            print("Failed to get enhanced response")
        print()
        return True


def main():
    parser = argparse.ArgumentParser(description="Docs2DB RAG Feature Demo")
    parser.add_argument(
        "--query",
        default="How do I configure SSH key-based authentication?",
        help="Query to search for (default: SSH configuration question)",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8321",
        help="Llama Stack server URL (default: http://localhost:8321)",
    )

    args = parser.parse_args()

    # Create and run demo client
    demo_client = Docs2DBRAGDemoClient(base_url=args.server)
    success = demo_client.run_demo(args.query)

    if not success:
        print("\nDemo failed. Check server status and try again.")
        sys.exit(1)

    print("\nDemo complete")


if __name__ == "__main__":
    main()
