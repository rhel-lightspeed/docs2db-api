#!/usr/bin/env python3
"""
OpenAI-Compatible RAG Tools
============================

OpenAI function calling compatible interface for the Universal RAG Engine.
This allows the RAG functionality to be used as tools/functions in OpenAI,
Anthropic, and other LLM providers that support function calling.

Features:
- OpenAI function calling schema
- Anthropic tool calling compatible
- Generic function calling interface
- Easy integration with AI frameworks
- Tool calling optimized responses

Usage:
    from codex.rag.openai_tools import get_openai_tools, call_rag_function

    # Get tool definitions for OpenAI
    tools = get_openai_tools()

    # Call function
    result = await call_rag_function("search_documents", {"query": "How to configure SSH?"})
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from codex.embeddings import EMBEDDING_CONFIGS
from codex.rag.engine import RAGConfig, UniversalRAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG engine instance
_rag_engine: Optional[UniversalRAGEngine] = None


async def get_rag_engine() -> UniversalRAGEngine:
    """Get or create the global RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        config = RAGConfig()  # Default configuration
        _rag_engine = UniversalRAGEngine(config)
        logger.info("Initialized global RAG engine for OpenAI tools")
    return _rag_engine


def get_openai_tools() -> List[Dict[str, Any]]:
    """
    Get OpenAI function calling tool definitions.

    Returns:
        List of OpenAI function calling tool definitions
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the RHEL knowledge base for relevant documents using advanced RAG techniques. Returns document chunks with similarity scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query or question to find relevant documents for",
                        },
                        "model_name": {
                            "type": "string",
                            "description": f"Embedding model to use for search (default: granite-30m-english)",
                            "enum": list(EMBEDDING_CONFIGS.keys()),
                            "default": "granite-30m-english",
                        },
                        "max_chunks": {
                            "type": "integer",
                            "description": "Maximum number of document chunks to return (1-50)",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10,
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold for results (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7,
                        },
                        "enable_question_refinement": {
                            "type": "boolean",
                            "description": "Enable question refinement for better retrieval results",
                            "default": True,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Alternative name for search_documents - search RHEL knowledge base for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query or question",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def get_anthropic_tools() -> List[Dict[str, Any]]:
    """
    Get Anthropic tool calling definitions.

    Returns:
        List of Anthropic tool definitions
    """
    return [
        {
            "name": "search_documents",
            "description": "Search the RHEL knowledge base for relevant documents using advanced RAG techniques. Returns document chunks with similarity scores and metadata.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to find relevant documents for",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Embedding model to use for search",
                        "enum": list(EMBEDDING_CONFIGS.keys()),
                        "default": "granite-30m-english",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum number of document chunks to return",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold for results (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                    },
                },
                "required": ["query"],
            },
        }
    ]


async def call_rag_function(
    function_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call a RAG function with the given arguments.

    Args:
        function_name: Name of the function to call
        arguments: Function arguments

    Returns:
        Function result in a format suitable for tool calling
    """
    try:
        rag_engine = await get_rag_engine()

        if function_name in ["search_documents", "search_knowledge_base"]:
            return await _handle_search_function(rag_engine, function_name, arguments)
        else:
            return {"error": f"Unknown function: {function_name}", "success": False}

    except Exception as e:
        logger.error(f"RAG function call failed: {e}")
        return {"error": f"Function execution failed: {str(e)}", "success": False}


async def _handle_search_function(
    rag_engine: UniversalRAGEngine, function_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle search function calls"""

    # Extract and validate query
    query = arguments.get("query", "")
    if not query:
        return {"error": "Query parameter is required", "success": False}

    logger.info(f"🔍 Tool calling search: {query[:100]}...")

    # Handle different function names and parameter styles
    if function_name == "search_knowledge_base":
        # Simple interface with just query and limit
        max_chunks = arguments.get("limit", 5)
        search_options = {
            "max_chunks": max_chunks,
            "similarity_threshold": 0.7,
            "enable_question_refinement": True,
        }
    else:
        # Full interface with all parameters
        search_options = {
            "model_name": arguments.get("model_name", "granite-30m-english"),
            "max_chunks": arguments.get("max_chunks", 10),
            "similarity_threshold": arguments.get("similarity_threshold", 0.7),
            "enable_question_refinement": arguments.get(
                "enable_question_refinement", True
            ),
            "enable_hybrid_search": arguments.get("enable_hybrid_search", True),
        }

    try:
        # Perform search
        result = await rag_engine.search_documents(query, **search_options)

        # Format response for tool calling
        documents = []
        for doc in result.documents:
            documents.append({
                "text": doc["text"],
                "similarity_score": round(doc["similarity_score"], 3),
                "source": doc["document_path"],
                "chunk_index": doc["chunk_index"],
            })

        # Create tool calling optimized response
        response = {
            "success": True,
            "query": query,
            "documents_found": len(documents),
            "documents": documents,
            "metadata": {
                "model_used": search_options.get("model_name", "granite-30m-english"),
                "similarity_threshold": search_options["similarity_threshold"],
                "question_refinement": bool(result.refined_questions),
                "search_features": result.metadata.get("features_used", [])
                if result.metadata
                else [],
            },
        }

        # Add refined questions if available
        if result.refined_questions:
            response["refined_questions"] = result.refined_questions

        # Add summary for easy consumption
        if documents:
            response["summary"] = (
                f"Found {len(documents)} relevant documents with similarity scores ranging from {min(d['similarity_score'] for d in documents):.3f} to {max(d['similarity_score'] for d in documents):.3f}"
            )
        else:
            response["summary"] = "No relevant documents found matching the query"

        logger.info(f"✅ Tool calling search completed: {len(documents)} documents")

        return response

    except Exception as e:
        logger.error(f"❌ Search function failed: {e}")
        return {"error": f"Search failed: {str(e)}", "success": False, "query": query}


class OpenAIRAGHandler:
    """
    Handler class for OpenAI function calling integration.
    Provides a clean interface for processing OpenAI function calls.
    """

    def __init__(self, model_name: str = "granite-30m-english"):
        self.model_name = model_name
        self.rag_engine = None

    async def initialize(self):
        """Initialize the RAG engine"""
        if self.rag_engine is None:
            config = RAGConfig(model_name=self.model_name)
            self.rag_engine = UniversalRAGEngine(config)
            logger.info(f"Initialized OpenAI RAG handler with model: {self.model_name}")

    async def handle_function_call(self, function_call: Dict[str, Any]) -> str:
        """
        Handle an OpenAI function call and return a string response.

        Args:
            function_call: OpenAI function call object with 'name' and 'arguments'

        Returns:
            JSON string response suitable for OpenAI function calling
        """
        await self.initialize()

        function_name = function_call.get("name", "")

        try:
            # Parse arguments (they come as JSON string from OpenAI)
            arguments_str = function_call.get("arguments", "{}")
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str

            # Call the function
            result = await call_rag_function(function_name, arguments)

            # Return JSON string
            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            error_result = {
                "error": f"Invalid JSON in function arguments: {str(e)}",
                "success": False,
            }
            return json.dumps(error_result)
        except Exception as e:
            error_result = {
                "error": f"Function call failed: {str(e)}",
                "success": False,
            }
            return json.dumps(error_result)

    async def close(self):
        """Clean up resources"""
        if self.rag_engine:
            await self.rag_engine.close()


# Convenience functions for different AI frameworks


async def openai_function_call(function_call: Dict[str, Any]) -> str:
    """
    Process an OpenAI function call.

    Args:
        function_call: OpenAI function call with 'name' and 'arguments'

    Returns:
        JSON string response
    """
    handler = OpenAIRAGHandler()
    try:
        return await handler.handle_function_call(function_call)
    finally:
        await handler.close()


async def anthropic_tool_call(
    tool_name: str, tool_input: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process an Anthropic tool call.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters

    Returns:
        Tool result dictionary
    """
    return await call_rag_function(tool_name, tool_input)


# Example usage and testing
async def test_openai_integration():
    """Test OpenAI function calling integration"""
    print("Testing OpenAI function calling integration...")

    # Get tool definitions
    tools = get_openai_tools()
    print(f"Available tools: {[tool['function']['name'] for tool in tools]}")

    # Test function call
    function_call = {
        "name": "search_documents",
        "arguments": json.dumps({
            "query": "How do I configure SSH on RHEL?",
            "max_chunks": 3,
        }),
    }

    result = await openai_function_call(function_call)
    print(f"Function result: {result[:500]}...")


async def test_anthropic_integration():
    """Test Anthropic tool calling integration"""
    print("Testing Anthropic tool calling integration...")

    # Get tool definitions
    tools = get_anthropic_tools()
    print(f"Available tools: {[tool['name'] for tool in tools]}")

    # Test tool call
    result = await anthropic_tool_call(
        "search_documents",
        {"query": "What are the system requirements for RHEL?", "max_chunks": 3},
    )

    print(f"Tool result: {json.dumps(result, indent=2)[:500]}...")


if __name__ == "__main__":

    async def main():
        await test_openai_integration()
        print("\n" + "=" * 50 + "\n")
        await test_anthropic_integration()

    asyncio.run(main())
