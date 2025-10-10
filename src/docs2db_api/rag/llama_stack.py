#!/usr/bin/env python3
"""
Llama Stack RAG Adapter
========================

Llama Stack Tool Runtime adapter that exposes the Universal RAG Engine
as built-in server-side tools for Llama Stack applications.

This adapter allows Llama Stack agents to use the advanced RAG capabilities
(question refinement, hybrid search, similarity post-processing) as native tools.

Features:
- Native Llama Stack tool integration
- Server-side RAG execution
- Compatible with existing Llama Stack clients
- Uses Docs2DB database and Granite embeddings
- Configurable model selection

Usage:
    # In Llama Stack configuration
    providers:
      tool_runtime:
        - provider_id: docs2db_rag
          provider_type: inline
          config:
            module: docs2db.rag.llama_stack
            config_class: Docs2DBRAGConfig
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

# Add threading safety for ML libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from docs2db_api.embeddings import EMBEDDING_CONFIGS
from docs2db_api.rag.engine import RAGConfig, UniversalRAGEngine, OllamaLLMClient

logger = structlog.get_logger(__name__)

# Set torch threading after import
try:
    import torch
    torch.set_num_threads(1)
    logger.info("🔧 Set PyTorch to single-threaded mode")
except ImportError:
    logger.info("🔧 PyTorch not available, skipping thread configuration")


@dataclass
class Docs2DBRAGConfig:
    """Configuration for Docs2DB RAG Provider"""

    model_name: str = "granite-30m-english"
    similarity_threshold: float = 0.7
    max_chunks: int = 10
    max_tokens_in_context: int = 4096
    enable_question_refinement: bool = True
    enable_hybrid_search: bool = True


# Try to import Llama Stack interfaces
try:
    from llama_stack.apis.tools import ToolInvocationResult, ToolRuntime
    from llama_stack.apis.tools.tools import (
        ListToolDefsResponse,
        ToolDef,
        ToolParameter,
    )

    LLAMA_STACK_AVAILABLE = True
except ImportError:
    logger.warning("Llama Stack not available - adapter will not function")
    LLAMA_STACK_AVAILABLE = False

    # Create dummy classes for development
    class ToolRuntime:
        pass

    class ToolInvocationResult:
        def __init__(
            self, content=None, error_message=None, error_code=None, metadata=None
        ):
            self.content = content
            self.error_message = error_message
            self.error_code = error_code
            self.metadata = metadata

    class ToolDef:
        def __init__(self, name, description, parameters):
            self.name = name
            self.description = description
            self.parameters = parameters

    class ToolParameter:
        def __init__(
            self, name, parameter_type, description, required=True, default=None
        ):
            self.name = name
            self.parameter_type = parameter_type
            self.description = description
            self.required = required
            self.default = default

    class ListToolDefsResponse:
        def __init__(self, data):
            self.data = data


class Docs2DBRAGAdapter(ToolRuntime):
    """
    Docs2DB RAG Tool Runtime Adapter for Llama Stack

    This adapter wraps the Universal RAG Engine to make it available
    as built-in Llama Stack tools for server-side RAG execution.
    """

    def __init__(self, config: Docs2DBRAGConfig):
        self.config = config
        self.rag_engine = None
        self._initialized = False

        logger.info(f"Initialized Docs2DB RAG adapter with model: {config.model_name}")

    async def _initialize_rag_engine(self):
        """Initialize the RAG engine (lazy loading)"""
        if self._initialized:
            return

        try:
            logger.info("🔧 Creating RAG configuration...")
            # Create RAG configuration
            rag_config = RAGConfig(
                model_name=self.config.model_name,
                similarity_threshold=self.config.similarity_threshold,
                max_chunks=self.config.max_chunks,
                max_tokens_in_context=self.config.max_tokens_in_context,
                enable_question_refinement=self.config.enable_question_refinement,
                enable_hybrid_search=self.config.enable_hybrid_search,
            )
            logger.info("✅ RAG configuration created")

            logger.info("🔧 Initializing Universal RAG Engine...")
            
            # Create Ollama LLM client for query refinement
            ollama_client = None
            if rag_config.enable_question_refinement:
                try:
                    ollama_client = OllamaLLMClient()
                    logger.info("✅ Ollama LLM client created for query refinement")
                except Exception as e:
                    logger.warning(f"Failed to create Ollama client, query refinement disabled: {e}")
            
            # Initialize the RAG engine with LLM client
            self.rag_engine = UniversalRAGEngine(rag_config, ollama_client)
            logger.info("✅ Universal RAG Engine created")

            self._initialized = True
            logger.info("✅ RAG engine initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG engine: {e}")
            raise e

    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[str] = None
    ) -> ListToolDefsResponse:
        """
        List all available runtime tools provided by this adapter.

        Returns:
            ListToolDefsResponse containing RAG tool definitions
        """
        # Document search tool parameters
        search_params = [
            ToolParameter(
                name="query",
                parameter_type="string",
                description="The search query or question",
                required=True,
            ),
            ToolParameter(
                name="model_name",
                parameter_type="string",
                description=f"Embedding model to use (default: {self.config.model_name})",
                required=False,
                default=self.config.model_name,
            ),
            ToolParameter(
                name="max_chunks",
                parameter_type="integer",
                description="Maximum number of document chunks to retrieve",
                required=False,
                default=self.config.max_chunks,
            ),
            ToolParameter(
                name="similarity_threshold",
                parameter_type="number",
                description="Minimum similarity threshold (0.0-1.0)",
                required=False,
                default=self.config.similarity_threshold,
            ),
            ToolParameter(
                name="enable_question_refinement",
                parameter_type="boolean",
                description="Enable question refinement for better retrieval",
                required=False,
                default=self.config.enable_question_refinement,
            ),
        ]

        # Response generation tool parameters (same as search + response generation)
        generate_params = search_params.copy()

        # Create tool definitions
        tools = [
            ToolDef(
                name="search_documents",
                description="Search RHEL knowledge base using advanced RAG techniques. Returns relevant document chunks with similarity scores.",
                parameters=search_params,
            ),
            ToolDef(
                name="search_and_generate",
                description="Search RHEL knowledge base and generate a comprehensive response using retrieved documents.",
                parameters=generate_params,
            ),
        ]

        return ListToolDefsResponse(data=tools)

    async def register_toolgroup(self, toolgroup: Dict[str, Any]) -> None:
        """
        Register a toolgroup with this provider.

        Args:
            toolgroup: The toolgroup configuration to register
        """
        logger.info(f"Registering toolgroup: {toolgroup}")
        # For inline providers, toolgroups are registered automatically
        pass

    async def invoke_tool(
        self, tool_name: str, kwargs: Dict[str, Any]
    ) -> ToolInvocationResult:
        """
        Invoke a RAG tool with given arguments

        Args:
            tool_name: Name of the tool ("search_documents" or "search_and_generate")
            kwargs: Tool arguments including query, model_name, etc.

        Returns:
            ToolInvocationResult with RAG response and metadata
        """
        try:
            # Ensure RAG engine is initialized
            await self._initialize_rag_engine()

            if tool_name == "search_documents":
                return await self._handle_search_documents(kwargs)
            elif tool_name == "search_and_generate":
                return await self._handle_search_and_generate(kwargs)
            else:
                return ToolInvocationResult(
                    content=None,
                    error_message=f"Unknown tool: {tool_name}",
                    error_code=404,
                )

        except Exception as e:
            logger.error(f"Error in tool invocation: {e}")
            return ToolInvocationResult(
                content=None,
                error_message=f"Tool execution failed: {str(e)}",
                error_code=500,
            )

    async def _handle_search_documents(
        self, kwargs: Dict[str, Any]
    ) -> ToolInvocationResult:
        """Handle document search requests"""

        # Extract and validate arguments
        query = kwargs.get("query", "")
        if not query:
            return ToolInvocationResult(
                content=None,
                error_message="Query parameter is required",
                error_code=400,
            )

        # Extract optional parameters with defaults
        model_name = kwargs.get("model_name", self.config.model_name)
        max_chunks = kwargs.get("max_chunks", self.config.max_chunks)
        similarity_threshold = kwargs.get(
            "similarity_threshold", self.config.similarity_threshold
        )
        enable_question_refinement = kwargs.get(
            "enable_question_refinement", self.config.enable_question_refinement
        )

        logger.info(
            f"🔍 Processing document search: {query[:100]}... (model: {model_name})"
        )

        try:
            logger.info("🔧 About to call rag_engine.search_documents...")
            # Perform document search
            result = await self.rag_engine.search_documents(
                query,
                model_name=model_name,
                max_chunks=max_chunks,
                similarity_threshold=similarity_threshold,
                enable_question_refinement=enable_question_refinement,
            )
            logger.info("✅ rag_engine.search_documents completed")

            # Format response for Llama Stack
            documents_summary = f"Found {len(result.documents)} relevant documents:\n\n"

            for i, doc in enumerate(result.documents, 1):
                documents_summary += (
                    f"Document {i} (similarity: {doc['similarity_score']:.3f}):\n"
                )
                documents_summary += (
                    f"{doc['text'][:500]}{'...' if len(doc['text']) > 500 else ''}\n\n"
                )

            # Create comprehensive metadata
            metadata = result.metadata or {}
            metadata.update({
                "tool_name": "search_documents",
                "documents_count": len(result.documents),
                "query_original": query,
                "refined_questions": result.refined_questions,
                "documents_details": [
                    {
                        "text_preview": doc["text"][:200] + "..."
                        if len(doc["text"]) > 200
                        else doc["text"],
                        "similarity_score": doc["similarity_score"],
                        "document_path": doc["document_path"],
                        "chunk_index": doc["chunk_index"],
                    }
                    for doc in result.documents
                ],
            })

            logger.info(
                f"✅ Document search completed - {len(result.documents)} documents found"
            )

            return ToolInvocationResult(content=documents_summary, metadata=metadata)

        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return ToolInvocationResult(
                content=None,
                error_message=f"Document search failed: {str(e)}",
                error_code=500,
            )

    async def _handle_search_and_generate(
        self, kwargs: Dict[str, Any]
    ) -> ToolInvocationResult:
        """Handle search + response generation requests"""

        # Extract and validate arguments
        query = kwargs.get("query", "")
        if not query:
            return ToolInvocationResult(
                content=None,
                error_message="Query parameter is required",
                error_code=400,
            )

        # Extract optional parameters
        model_name = kwargs.get("model_name", self.config.model_name)
        max_chunks = kwargs.get("max_chunks", self.config.max_chunks)
        similarity_threshold = kwargs.get(
            "similarity_threshold", self.config.similarity_threshold
        )
        enable_question_refinement = kwargs.get(
            "enable_question_refinement", self.config.enable_question_refinement
        )

        logger.info(
            f"🚀 Processing search and generate: {query[:100]}... (model: {model_name})"
        )

        try:
            # For now, fall back to document search since we don't have LLM integration yet
            # TODO: Implement full search_and_generate when LLM client is available
            result = await self.rag_engine.search_documents(
                query,
                model_name=model_name,
                max_chunks=max_chunks,
                similarity_threshold=similarity_threshold,
                enable_question_refinement=enable_question_refinement,
            )

            # Create a summary response based on retrieved documents
            if not result.documents:
                response_content = (
                    "I couldn't find relevant information to answer your question."
                )
            else:
                response_content = f"Based on {len(result.documents)} relevant documents from the RHEL knowledge base:\n\n"

                # Include top documents in response
                for i, doc in enumerate(result.documents[:3], 1):  # Top 3 documents
                    response_content += f"Document {i}: {doc['text'][:300]}{'...' if len(doc['text']) > 300 else ''}\n\n"

                response_content += f"(Note: Full response generation requires LLM integration. Found {len(result.documents)} total relevant documents.)"

            # Create metadata
            metadata = result.metadata or {}
            metadata.update({
                "tool_name": "search_and_generate",
                "documents_count": len(result.documents),
                "query_original": query,
                "refined_questions": result.refined_questions,
                "generation_status": "fallback_to_search_only",
                "documents_used": len(result.documents),
            })

            logger.info(
                f"✅ Search and generate completed - {len(result.documents)} documents processed"
            )

            return ToolInvocationResult(content=response_content, metadata=metadata)

        except Exception as e:
            logger.error(f"❌ Search and generate failed: {e}")
            return ToolInvocationResult(
                content=None,
                error_message=f"Search and generate failed: {str(e)}",
                error_code=500,
            )


# Required function for inline providers
async def get_provider_impl(
    config: Docs2DBRAGConfig, deps: Dict[Any, Any]
) -> Docs2DBRAGAdapter:
    """
    Required function for Llama Stack inline providers.

    Args:
        config: An instance of Docs2DBRAGConfig
        deps: A dictionary of API dependencies

    Returns:
        An instance of Docs2DBRAGAdapter
    """
    logger.info("Creating Docs2DB RAG provider instance")
    adapter = Docs2DBRAGAdapter(config)
    return adapter


# For testing the provider independently
async def test_provider():
    """Test the provider functionality"""
    config = Docs2DBRAGConfig(
        model_name="granite-30m-english", similarity_threshold=0.7, max_chunks=5
    )

    adapter = await get_provider_impl(config, {})

    # Test document search
    search_kwargs = {"query": "How do I configure SSH on RHEL?", "max_chunks": 3}

    search_response = await adapter.invoke_tool("search_documents", search_kwargs)
    logger.info("Search Response", content_preview=search_response.content[:200])

    # Test search and generate
    generate_response = await adapter.invoke_tool("search_and_generate", search_kwargs)
    logger.info("Generate Response", content_preview=generate_response.content[:200])


if __name__ == "__main__":
    asyncio.run(test_provider())
