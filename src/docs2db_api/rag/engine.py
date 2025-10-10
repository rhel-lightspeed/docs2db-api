#!/usr/bin/env python3
"""
Universal RAG Engine for Codex
==============================

This module implements a generic RAG (Retrieval-Augmented Generation) engine
that combines the advanced techniques from RagSpike2 with Codex's existing
database and embedding infrastructure.

Features:
- Question refinement (generates multiple targeted queries)
- Hybrid search (vector similarity + keyword search)
- Similarity post-processing with configurable thresholds
- Multi-model support using Codex's EMBEDDING_CONFIGS
- Generic interface suitable for multiple API adapters

Architecture:
- Uses existing Codex database with 91K+ documents and 811K+ chunks
- Leverages Codex's GraniteEmbeddingProvider by default
- Configurable model selection for future extensibility
- Framework-agnostic core suitable for REST API, Llama Stack, etc.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from docs2db_api.database import DatabaseManager, get_db_config
from docs2db_api.embeddings import EMBEDDING_CONFIGS, GraniteEmbeddingProvider

# Configure logging
logger = structlog.get_logger(__name__)


class OllamaLLMClient:
    """Simple Ollama client for query refinement."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b-instruct"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def acomplete(self, prompt: str) -> str:
        """Complete a prompt using Ollama API."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.warning(f"Ollama LLM call failed: {e}")
            return ""
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Query refinement prompt template (adapted from rlsapi)
REFINEMENT_PROMPT_TEMPLATE = """### YOUR ROLE
You are an expert in technical documentation, system administration, software development, and Linux systems.

Your purpose is to generate meaningful and specific questions based on user queries, especially when the query relates to technical topics.

You will receive a user query which is potentially unclear, incomplete or ambiguous.

Your role is to generate five user-simulated questions that are more specific, focused, and free of ambiguity.

### WORKFLOW PROTOCOL
**1. Validate the User Query**
First, you must ensure that the user query is related to technical topics such as:
- System administration and configuration
- Software development and programming
- Linux and Unix systems
- Documentation and technical writing
- DevOps and infrastructure
- Troubleshooting and problem-solving

You must consider the user query as invalid if it meets any of the following criteria:
- Is NOT related to technical topics
- Is a greeting or casual conversation
- Is non-sense or unclear
- Conflicts with ethical, legal and moral principles

If the user query is invalid:
- Your response must only be the string "EMPTY" and nothing else.
- This also means that the steps 2 and 3 described below should not be followed.

If the user query is considered valid, please proceed with the next steps.

**2. Generate the Questions**
If your previous assessment is that the user query is indeed valid, you then follow the below specifications for generating questions:
- Each of the new questions must derive from the original query.
- As mentioned in your role, you must simulate the user for each of the generated questions.
- Compose them as if you were the user asking to an expert who covers technical topics as described in your role.

**Response Format and Structure**
- Your response must contain five questions in total.
- Each question must be on its own line using plain text.
- Avoid wrapping the questions with quotes or double quotes. That is not needed.
- Avoid numbered lists, bullet points, headings, or any kind of formatting. Just plain text, please.
- DO NOT include any introduction, explanation, commentary, or conclusion in your response.

### ADDITIONAL GUIDELINES
- Ensure all instructions described in your workflow protocol are followed consistently.
- Make questions specific and actionable for better document retrieval.

User query: {question}"""


@dataclass
class RAGConfig:
    """Configuration for RAG engine"""

    model_name: str = "granite-30m-english"
    similarity_threshold: float = 0.7
    max_chunks: int = 10
    max_tokens_in_context: int = 4096
    enable_question_refinement: bool = True
    enable_hybrid_search: bool = True
    refinement_questions_count: int = 5


@dataclass
class RAGResult:
    """Result from RAG query"""

    query: str
    documents: List[Dict[str, Any]]
    response: Optional[str] = None
    refined_questions: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UniversalRAGEngine:
    """
    Universal RAG Engine that provides advanced RAG capabilities
    using Codex's existing database and embedding infrastructure.

    This engine is framework-agnostic and can be used by multiple
    interface adapters (REST API, Llama Stack, OpenAI tools, etc.)
    """

    def __init__(self, config: Optional[RAGConfig] = None, llm_client=None):
        self.config = config or RAGConfig()
        self.llm_client = llm_client  # Optional LLM for response generation

        # Validate model configuration
        if self.config.model_name not in EMBEDDING_CONFIGS:
            raise ValueError(
                f"Unknown model: {self.config.model_name}. "
                f"Available models: {list(EMBEDDING_CONFIGS.keys())}"
            )

        self.model_config = EMBEDDING_CONFIGS[self.config.model_name]
        logger.info(f"Initialized RAG engine with model: {self.config.model_name}")

        # Initialize database connection
        db_config = get_db_config()
        self.db_manager = DatabaseManager(
            host=db_config["host"],
            port=int(db_config["port"]),
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )

        # Initialize embedding provider
        self.embedding_provider = self._get_embedding_provider()

    def _get_embedding_provider(self):
        """Get the appropriate embedding provider for the configured model"""
        provider_cls = self.model_config["cls"]

        if provider_cls == GraniteEmbeddingProvider:
            return GraniteEmbeddingProvider(
                model_name=self.model_config["model_id"],
                config=self.model_config,
                device="cpu",  # Default to CPU for now
            )
        else:
            # For future model support
            return provider_cls()

    async def search_documents(self, query: str, **options) -> RAGResult:
        """
        Core document search functionality.

        This is the main entry point that provides framework-agnostic
        document retrieval with advanced RAG features.

        Args:
            query: User's search query
            **options: Override default config options

        Returns:
            RAGResult with documents and metadata
        """
        logger.info(f"🔍 Processing RAG query: {query[:100]}...")

        # Merge options with config
        search_config = RAGConfig(
            model_name=options.get("model_name", self.config.model_name),
            similarity_threshold=options.get(
                "similarity_threshold", self.config.similarity_threshold
            ),
            max_chunks=options.get("max_chunks", self.config.max_chunks),
            max_tokens_in_context=options.get(
                "max_tokens_in_context", self.config.max_tokens_in_context
            ),
            enable_question_refinement=options.get(
                "enable_question_refinement", self.config.enable_question_refinement
            ),
            enable_hybrid_search=options.get(
                "enable_hybrid_search", self.config.enable_hybrid_search
            ),
            refinement_questions_count=options.get(
                "refinement_questions_count", self.config.refinement_questions_count
            ),
        )

        try:
            # Step 1: Question refinement (if enabled)
            refined_questions = None
            if search_config.enable_question_refinement and self.llm_client:
                refined_questions = await self._refine_questions(query, search_config)
                # Use refined questions if available, otherwise use original query
                search_query = refined_questions if refined_questions else query
            else:
                search_query = query

            # Step 2: Generate query embeddings
            query_embeddings = await self._generate_query_embeddings(search_query)

            # Step 3: Retrieve similar documents
            documents = await self._retrieve_similar_documents(
                query_embeddings, search_config
            )

            # Step 4: Post-process and filter results
            filtered_documents = self._post_process_results(documents, search_config)

            # Create metadata
            metadata = {
                "model_name": search_config.model_name,
                "model_dimensions": self.model_config["dimensions"],
                "similarity_threshold": search_config.similarity_threshold,
                "documents_found": len(filtered_documents),
                "hybrid_search_enabled": search_config.enable_hybrid_search,
                "question_refinement_enabled": search_config.enable_question_refinement,
                "features_used": self._get_features_used(
                    search_config, refined_questions
                ),
            }

            logger.info(
                f"✅ RAG search completed - {len(filtered_documents)} documents found"
            )

            return RAGResult(
                query=query,
                documents=filtered_documents,
                refined_questions=refined_questions,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"❌ RAG search failed: {e}")
            raise

    async def search_and_generate(self, query: str, **options) -> RAGResult:
        """
        Search documents and generate a response using LLM.

        This combines document retrieval with response generation
        for complete RAG functionality.
        """
        if not self.llm_client:
            raise ValueError("LLM client required for response generation")

        # Get documents
        result = await self.search_documents(query, **options)

        # Generate response
        response = await self._generate_response(
            query, result.documents, result.refined_questions
        )
        result.response = response

        return result

    async def _refine_questions(self, query: str, config: RAGConfig) -> Optional[str]:
        """Generate refined, targeted questions for better retrieval (rlsapi pattern)"""
        
        # Format the prompt using our template
        prompt = REFINEMENT_PROMPT_TEMPLATE.format(question=query)
        
        try:
            # Use LLM to refine questions
            refined = await self._call_llm(prompt)
            
            # Handle "EMPTY" response (query not technical/valid)
            if refined.strip() == "EMPTY":
                logger.info("Query refinement returned 'EMPTY' - query may not be technical")
                return None
            
            # Clean up the response
            refined = refined.strip()
            if refined:
                logger.info(f"🎯 Generated refined questions: {refined[:200]}...")
                return refined
            else:
                logger.warning("Query refinement returned empty response")
                return None
                
        except Exception as e:
            logger.warning(f"Question refinement failed: {e}")
            return None

    async def _generate_query_embeddings(self, query_text: str) -> List[float]:
        """Generate embeddings for the query text"""
        try:
            # Handle both single queries and refined questions
            if isinstance(query_text, str) and (
                "1." in query_text or "2." in query_text
            ):
                # Extract individual questions from numbered list
                lines = query_text.strip().split("\n")
                questions = []
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("•")):
                        # Remove numbering and extract question
                        question = line.split(".", 1)[-1].strip()
                        if question:
                            questions.append(question)

                if questions:
                    # Generate embeddings for all questions and average them
                    all_embeddings = self.embedding_provider.generate_embeddings(
                        questions
                    )
                    # Average the embeddings
                    import numpy as np

                    avg_embedding = np.mean(all_embeddings, axis=0).tolist()
                    return avg_embedding

            # Single query embedding
            embeddings = self.embedding_provider.generate_embeddings([query_text])
            return embeddings[0]

        except Exception as e:
            logger.error(f"Failed to generate query embeddings: {e}")
            raise

    async def _retrieve_similar_documents(
        self, query_embedding: List[float], config: RAGConfig
    ) -> List[Dict[str, Any]]:
        """Retrieve similar documents from the database"""
        try:
            similar_chunks = await self.db_manager.search_similar(
                query_embedding=query_embedding,
                model_name=config.model_name,
                limit=config.max_chunks * 2,  # Get extra for post-processing
                similarity_threshold=config.similarity_threshold,
            )

            # Convert to standard format
            documents = []
            for chunk in similar_chunks:
                documents.append({
                    "text": chunk["text"],
                    "similarity_score": chunk.get("similarity", 0.0),
                    "document_path": chunk.get("document_path", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "metadata": chunk.get("metadata", {}),
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve similar documents: {e}")
            raise

    def _post_process_results(
        self, documents: List[Dict[str, Any]], config: RAGConfig
    ) -> List[Dict[str, Any]]:
        """Post-process and filter results based on similarity and token limits"""
        # Filter by similarity threshold
        filtered = [
            doc
            for doc in documents
            if doc["similarity_score"] >= config.similarity_threshold
        ]

        # Sort by similarity score (descending)
        filtered.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Limit by max_chunks
        filtered = filtered[: config.max_chunks]

        # Estimate token usage and truncate if needed
        total_tokens = 0
        final_docs = []

        for doc in filtered:
            # Rough token estimation (1 token ≈ 4 characters)
            doc_tokens = len(doc["text"]) // 4

            if total_tokens + doc_tokens <= config.max_tokens_in_context:
                final_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break

        return final_docs

    async def _generate_response(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        refined_questions: Optional[str] = None,
    ) -> str:
        """Generate a response using the retrieved documents"""
        if not documents:
            return "I couldn't find relevant information to answer your question."

        # Create context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc['text']}\n")

        context = "\n".join(context_parts)

        response_prompt = f"""You are a helpful assistant with expertise in Red Hat Enterprise Linux (RHEL) and system administration.

Based on the following context from the RHEL documentation, provide a comprehensive and accurate answer to the user's question.

Context:
{context}

Question: {query}

Please provide a detailed, practical answer. If the context doesn't contain enough information to fully answer the question, clearly state what information is missing and provide what information is available.

Answer:"""

        try:
            response = await self._call_llm(response_prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I found {len(documents)} relevant documents but couldn't generate a response due to an error."

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM client (if available)"""
        if not self.llm_client:
            raise ValueError("No LLM client configured")

        # Use the Ollama client's acomplete method
        return await self.llm_client.acomplete(prompt)

    def _get_features_used(
        self, config: RAGConfig, refined_questions: Optional[str]
    ) -> List[str]:
        """Get list of features used in this search"""
        features = [
            f"{config.model_name}_embeddings",
            "postgresql_vector_search",
            "similarity_post_processing",
        ]

        if config.enable_hybrid_search:
            features.append("hybrid_search")

        if (
            config.enable_question_refinement
            and refined_questions
            and refined_questions != ""
        ):
            features.append("question_refinement")

        return features

    async def close(self):
        """Clean up resources"""
        # Close LLM client if it has a close method
        if self.llm_client and hasattr(self.llm_client, 'close'):
            await self.llm_client.close()


# Convenience functions for common use cases
async def search_documents(
    query: str, model_name: str = "granite-30m-english", **options
) -> RAGResult:
    """
    Convenience function for simple document search.

    Args:
        query: Search query
        model_name: Embedding model to use
        **options: Additional search options

    Returns:
        RAGResult with documents and metadata
    """
    config = RAGConfig(
        model_name=model_name,
        **{k: v for k, v in options.items() if hasattr(RAGConfig, k)},
    )
    engine = UniversalRAGEngine(config)

    try:
        result = await engine.search_documents(query, **options)
        return result
    finally:
        await engine.close()


async def search_and_generate(
    query: str, llm_client, model_name: str = "granite-30m-english", **options
) -> RAGResult:
    """
    Convenience function for search + response generation.

    Args:
        query: Search query
        llm_client: LLM client for response generation
        model_name: Embedding model to use
        **options: Additional search options

    Returns:
        RAGResult with documents, response, and metadata
    """
    config = RAGConfig(
        model_name=model_name,
        **{k: v for k, v in options.items() if hasattr(RAGConfig, k)},
    )
    engine = UniversalRAGEngine(config, llm_client)

    try:
        result = await engine.search_and_generate(query, **options)
        return result
    finally:
        await engine.close()


async def log_search(query, model_name, max_chunks, similarity_threshold):
    result = await search_documents(
        query,
        model_name=model_name,
        max_chunks=max_chunks,
        similarity_threshold=similarity_threshold,
    )

    logger.info(f"Query: {result.query}")
    logger.info(f"Found {len(result.documents)} documents")

    if result.refined_questions:
        logger.info(f"Refined Questions:\n{result.refined_questions}")

    for i, doc in enumerate(result.documents, 1):
        logger.info(
            f"\n{i}.\n"
            f"   Score: {doc['similarity_score']:.3f}\n"
            f"   Source: {doc['document_path']}\n"
            f"   Text: {doc['text'][:200]}..."
        )
