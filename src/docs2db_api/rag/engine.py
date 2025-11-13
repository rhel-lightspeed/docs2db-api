#!/usr/bin/env python3
"""
Universal RAG Engine
====================

Features:
- Question refinement (generates multiple targeted queries)
- Hybrid search (vector similarity + keyword search)
- Similarity post-processing with configurable thresholds
- Multi-model support
- Generic interface suitable for multiple API adapters

Architecture:
- Uses Docs2DB databases (https://github.com/rhel-lightspeed/docs2db)
- Configurable model selection for future extensibility
- Framework-agnostic core suitable for REST API, Llama Stack, etc.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast, overload

import httpx
import structlog

from docs2db_api.database import DatabaseManager, get_db_config
from docs2db_api.embeddings import EMBEDDING_CONFIGS, GraniteEmbeddingProvider
from docs2db_api.reranker import get_reranker

# Configure logging
logger = structlog.get_logger(__name__)

# Code defaults for RAG settings (lowest priority in hierarchy)
DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
DEFAULT_MAX_CHUNKS: int = 10
DEFAULT_MAX_TOKENS_IN_CONTEXT: int = 4096
DEFAULT_ENABLE_QUESTION_REFINEMENT: bool = True
DEFAULT_ENABLE_RERANKING: bool = True
DEFAULT_REFINEMENT_QUESTIONS_COUNT: int = 5

DEFAULT_RAG_SETTINGS: Dict[str, Union[float, int, bool]] = {
    "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
    "max_chunks": DEFAULT_MAX_CHUNKS,
    "max_tokens_in_context": DEFAULT_MAX_TOKENS_IN_CONTEXT,
    "enable_question_refinement": DEFAULT_ENABLE_QUESTION_REFINEMENT,
    "enable_reranking": DEFAULT_ENABLE_RERANKING,
    "refinement_questions_count": DEFAULT_REFINEMENT_QUESTIONS_COUNT,
}


# Type-safe setting getter with overloads
@overload
def _get_setting(
    config_value: Optional[bool],
    env_var: str,
    db_value: Optional[bool],
    default_value: bool,
    setting_type: type[bool],
    logger: Any
) -> bool: ...

@overload
def _get_setting(
    config_value: Optional[int],
    env_var: str,
    db_value: Optional[int],
    default_value: int,
    setting_type: type[int],
    logger: Any
) -> int: ...

@overload
def _get_setting(
    config_value: Optional[float],
    env_var: str,
    db_value: Optional[float],
    default_value: float,
    setting_type: type[float],
    logger: Any
) -> float: ...

@overload
def _get_setting(
    config_value: Optional[str],
    env_var: str,
    db_value: Optional[str],
    default_value: str,
    setting_type: type[str],
    logger: Any
) -> str: ...

def _get_setting(
    config_value: Any,
    env_var: str,
    db_value: Any,
    default_value: Any,
    setting_type: type,
    logger: Any
) -> Any:
    """Get setting value following hierarchy: CLI/kwargs → env → database → defaults."""
    import os
    
    # 1. CLI/kwargs (config value)
    if config_value is not None:
        return config_value
    
    # 2. Environment variable
    env_value = os.getenv(env_var)
    if env_value is not None:
        try:
            if setting_type == bool:
                return env_value.lower() in ("true", "1", "yes")
            elif setting_type == int:
                return int(env_value)
            elif setting_type == float:
                return float(env_value)
            else:
                return env_value
        except (ValueError, AttributeError):
            logger.warning(f"Invalid environment variable {env_var}={env_value}, ignoring")
    
    # 3. Database value
    if db_value is not None:
        return db_value
    
    # 4. Code default
    return default_value


class LLMClient:
    """LLM client using OpenAI-compatible API for query refinement."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b-instruct"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def acomplete(self, prompt: str) -> str:
        """Complete a prompt using OpenAI-compatible API."""
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.7,
                    "max_tokens": 500,
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Query refinement prompt template for RAG
REFINEMENT_PROMPT_TEMPLATE = """### YOUR ROLE
You are an expert research assistant helping users find relevant information from a document collection.

Your purpose is to generate meaningful and specific questions based on user queries that may be unclear, incomplete, or ambiguous.

Your role is to generate five refined questions that are more specific, focused, and free of ambiguity to improve document retrieval.

### WORKFLOW PROTOCOL
**1. Validate the User Query**
You must consider the user query as invalid if it meets any of the following criteria:
- Is a greeting or casual conversation (e.g., "hello", "hi there", "how are you")
- Is non-sense, gibberish, or completely unclear
- Contains only a single word without context
- Is empty or has no meaningful content
- Conflicts with ethical, legal, or moral principles

If the user query is invalid:
- Your response must only be the string "EMPTY" and nothing else.
- Do not proceed with generating questions.

If the user query appears to be a genuine information-seeking question, proceed with the next steps.

**2. Generate the Questions**
Generate five refined questions following these specifications:
- Each question must derive from and relate to the original query
- Rephrase the query from different angles or perspectives
- Add specificity where the original query is vague
- Break down complex queries into focused sub-questions
- Consider different interpretations of ambiguous queries
- Make questions suitable for retrieving relevant documents

**3. Response Format and Structure**
- Your response must contain exactly five questions
- Each question must be on its own line using plain text
- Do NOT wrap questions with quotes or double quotes
- Do NOT use numbered lists, bullet points, headings, or any formatting
- Do NOT include any introduction, explanation, commentary, or conclusion
- Just five plain text questions, one per line

### ADDITIONAL GUIDELINES
- Ensure all instructions are followed consistently
- Make questions specific and actionable for document retrieval
- Maintain the intent and scope of the original query
- Use clear, natural language

User query: {question}"""


@dataclass
class RAGConfig:
    """Configuration for RAG engine
    
    All settings are optional (None = fall through to next level in hierarchy):
    CLI/kwargs → environment → .env → database → code defaults
    """

    model_name: Optional[str] = None  # If None, will auto-detect from database
    similarity_threshold: Optional[float] = None
    max_chunks: Optional[int] = None
    max_tokens_in_context: Optional[int] = None
    enable_question_refinement: Optional[bool] = None
    enable_reranking: Optional[bool] = None
    refinement_questions_count: Optional[int] = None


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
    This engine is framework-agnostic and can be used by multiple
    interface adapters (REST API, Llama Stack, OpenAI tools, etc.)
    
    Uses two-phase initialization pattern:
    - Constructor is lightweight, doesn't touch resources
    - start() method initializes database connection and detects model
    
    Usage:
        engine = UniversalRAGEngine(config=config, db_config=db_config)
        await engine.start()
        result = await engine.search_documents(query)
        
        # Note: No cleanup needed for typical usage (database connections are 
        # per-request). Only call close() if you provided an llm_client that 
        # needs cleanup.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm_client=None,
        db_config: Optional[Dict[str, str]] = None,
        refinement_prompt: Optional[str] = None,
    ):
        """Initialize RAG engine (lightweight, no I/O).
        
        Args:
            config: RAG configuration. If model_name is None, will auto-detect from database.
            llm_client: Optional LLM client for question refinement.
            db_config: Database configuration dict. If None, will auto-detect.
            refinement_prompt: Custom prompt for query refinement. If None, uses default or database value.
        """
        self.config = config or RAGConfig()
        self.llm_client = llm_client
        self._db_config_dict = db_config
        self.refinement_prompt = refinement_prompt
        
        # These will be initialized in start()
        self.db_manager: Optional[DatabaseManager] = None
        self.embedding_provider = None
        self.model_config: Optional[Dict[str, Any]] = None
        self._started = False

    async def start(self) -> None:
        """Initialize database connection and auto-detect model if needed.
        
        This method performs I/O and can fail if database is unavailable.
        Must be called before using the engine.
        
        Raises:
            ValueError: If no models found in database or model_name is invalid
            DatabaseError: If database connection fails
        """
        if self._started:
            logger.warning("RAG engine already started, ignoring duplicate start() call")
            return

        # Initialize database connection
        if self._db_config_dict:
            logger.info("Using provided database configuration")
            self.db_manager = DatabaseManager(
                host=self._db_config_dict["host"],
                port=int(self._db_config_dict["port"]),
                database=self._db_config_dict["database"],
                user=self._db_config_dict["user"],
                password=self._db_config_dict["password"],
            )
        else:
            logger.info("Auto-detecting database configuration")
            detected_config = get_db_config()
            self.db_manager = DatabaseManager(
                host=detected_config["host"],
                port=int(detected_config["port"]),
                database=detected_config["database"],
                user=detected_config["user"],
                password=detected_config["password"],
            )

        # Auto-detect model from database if not specified
        if self.config.model_name is None:
            logger.info("Model not specified, querying database for available models...")
            async with await self.db_manager.get_direct_connection() as conn:
                result = await conn.execute(
                    "SELECT name, dimensions, provider FROM models ORDER BY created_at DESC"
                )
                models = await result.fetchall()
                
                if not models:
                    raise ValueError(
                        "No embedding models found in database. "
                        "Please load documents first using docs2db CLI."
                    )
                
                if len(models) > 1:
                    model_names = [row[0] for row in models]
                    logger.warning(
                        f"Multiple models found in database: {model_names}. "
                        f"Using most recent: {models[0][0]}"
                    )
                
                # Use the most recently created model
                self.config.model_name = models[0][0]
                logger.info(
                    f"✅ Auto-detected model: {self.config.model_name} "
                    f"(dimensions: {models[0][1]}, provider: {models[0][2]})"
                )

        # Validate model configuration
        if self.config.model_name not in EMBEDDING_CONFIGS:
            raise ValueError(
                f"Unknown model: {self.config.model_name}. "
                f"Available models: {list(EMBEDDING_CONFIGS.keys())}"
            )

        self.model_config = EMBEDDING_CONFIGS[self.config.model_name]
        logger.info(f"Initialized RAG engine with model: {self.config.model_name}")

        # Apply settings hierarchy: CLI/kwargs → env → database → defaults
        await self._apply_settings_hierarchy()

        # Initialize embedding provider
        self.embedding_provider = self._get_embedding_provider()
        self._started = True
        
        # Assert all required config values are set after applying hierarchy
        assert self.config.model_name is not None, "model_name must be set after start()"
        assert self.config.similarity_threshold is not None, "similarity_threshold must be set"
        assert self.config.max_chunks is not None, "max_chunks must be set"
        assert self.config.max_tokens_in_context is not None, "max_tokens_in_context must be set"
        assert self.config.enable_question_refinement is not None, "enable_question_refinement must be set"
        assert self.config.enable_reranking is not None, "enable_reranking must be set"
        assert self.config.refinement_questions_count is not None, "refinement_questions_count must be set"

    async def _apply_settings_hierarchy(self) -> None:
        """Apply settings hierarchy: CLI/kwargs → env → database → defaults.
        
        For each setting:
        1. If explicitly set in config (not None), keep it (CLI/kwargs priority)
        2. Otherwise, check environment variable
        3. Otherwise, check database
        4. Otherwise, use code default
        """
        import os
        
        # Load database settings
        db_settings: Dict[str, Union[bool, int, float, None]] = {}
        db_refinement_prompt: Optional[str] = None
        
        assert self.db_manager is not None, "db_manager must be initialized"
        
        try:
            async with await self.db_manager.get_direct_connection() as conn:
                result = await conn.execute(
                    """
                    SELECT refinement_prompt, enable_refinement, enable_reranking,
                           similarity_threshold, max_chunks, max_tokens_in_context,
                           refinement_questions_count
                    FROM rag_settings WHERE id = 1
                    """
                )
                row = await result.fetchone()
                
                if row:
                    db_refinement_prompt = row[0]
                    db_settings = {
                        "enable_refinement": bool(row[1]) if row[1] is not None else None,
                        "enable_reranking": bool(row[2]) if row[2] is not None else None,
                        "similarity_threshold": float(row[3]) if row[3] is not None else None,
                        "max_chunks": int(row[4]) if row[4] is not None else None,
                        "max_tokens_in_context": int(row[5]) if row[5] is not None else None,
                        "refinement_questions_count": int(row[6]) if row[6] is not None else None,
                    }
                    logger.info("✅ Loaded RAG settings from database")
                else:
                    logger.info("No RAG settings found in database, using defaults")
        except Exception as e:
            logger.warning(f"Could not load RAG settings from database: {e}. Using defaults.")
        
        # Apply hierarchy for each setting
        # Priority: config (CLI/kwargs) → env → database → defaults
        
        # Apply to each config field
        self.config.similarity_threshold = _get_setting(
            self.config.similarity_threshold,
            "SIMILARITY_THRESHOLD",
            cast(Optional[float], db_settings.get("similarity_threshold")),
            DEFAULT_SIMILARITY_THRESHOLD,
            float,
            logger
        )
        
        self.config.max_chunks = _get_setting(
            self.config.max_chunks,
            "MAX_CHUNKS",
            cast(Optional[int], db_settings.get("max_chunks")),
            DEFAULT_MAX_CHUNKS,
            int,
            logger
        )
        
        self.config.max_tokens_in_context = _get_setting(
            self.config.max_tokens_in_context,
            "MAX_TOKENS_IN_CONTEXT",
            cast(Optional[int], db_settings.get("max_tokens_in_context")),
            DEFAULT_MAX_TOKENS_IN_CONTEXT,
            int,
            logger
        )
        
        self.config.enable_question_refinement = _get_setting(
            self.config.enable_question_refinement,
            "ENABLE_QUESTION_REFINEMENT",
            cast(Optional[bool], db_settings.get("enable_refinement")),
            DEFAULT_ENABLE_QUESTION_REFINEMENT,
            bool,
            logger
        )
        
        self.config.enable_reranking = _get_setting(
            self.config.enable_reranking,
            "ENABLE_RERANKING",
            cast(Optional[bool], db_settings.get("enable_reranking")),
            DEFAULT_ENABLE_RERANKING,
            bool,
            logger
        )
        
        self.config.refinement_questions_count = _get_setting(
            self.config.refinement_questions_count,
            "REFINEMENT_QUESTIONS_COUNT",
            cast(Optional[int], db_settings.get("refinement_questions_count")),
            DEFAULT_REFINEMENT_QUESTIONS_COUNT,
            int,
            logger
        )
        
        # Apply hierarchy for refinement prompt
        if self.refinement_prompt is None:
            env_prompt = os.getenv("REFINEMENT_PROMPT")
            if env_prompt:
                self.refinement_prompt = env_prompt
            elif db_refinement_prompt:
                self.refinement_prompt = db_refinement_prompt
            # Otherwise it stays None and will use default REFINEMENT_PROMPT_TEMPLATE
        
        logger.info(
            f"📋 Final RAG settings: threshold={self.config.similarity_threshold}, "
            f"max_chunks={self.config.max_chunks}, refinement={self.config.enable_question_refinement}, "
            f"reranking={self.config.enable_reranking}"
        )

    def _get_embedding_provider(self):
        """Get the appropriate embedding provider for the configured model"""
        assert self.model_config is not None, "model_config must be set before calling this method"
        
        provider_cls = self.model_config["cls"]

        if provider_cls == GraniteEmbeddingProvider:
            assert self.config.model_name is not None  # Set by start()
            return GraniteEmbeddingProvider(
                model_name=self.config.model_name,
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
            
        Raises:
            RuntimeError: If start() has not been called yet
        """
        if not self._started:
            raise RuntimeError(
                "RAG engine not initialized. Call await engine.start() first."
            )
        
        # Type checker assertions - these are guaranteed to be set after start()
        assert self.db_manager is not None
        assert self.embedding_provider is not None
        assert self.model_config is not None
        assert self.config.model_name is not None
        
        logger.info(f"Processing RAG query: {query[:100]}...")

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
            enable_reranking=options.get(
                "enable_reranking", self.config.enable_reranking
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

            # Step 3: Retrieve similar documents using hybrid search
            documents = await self._retrieve_similar_documents(
                query_embeddings, search_config, query
            )

            # Step 4: Rerank results with cross-encoder (if enabled)
            if documents and search_config.enable_reranking:
                documents = await self._rerank_documents(query, documents)

            # Step 5: Post-process and filter results
            filtered_documents = self._post_process_results(documents, search_config)

            # Create metadata
            metadata = {
                "model_name": search_config.model_name,
                "model_dimensions": self.model_config["dimensions"],
                "similarity_threshold": search_config.similarity_threshold,
                "documents_found": len(filtered_documents),
                "question_refinement_enabled": search_config.enable_question_refinement,
                "features_used": self._get_features_used(
                    search_config, refined_questions
                ),
            }

            logger.info(
                f"RAG search completed - {len(filtered_documents)} documents found"
            )

            return RAGResult(
                query=query,
                documents=filtered_documents,
                refined_questions=refined_questions,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            raise

    async def _refine_questions(self, query: str, config: RAGConfig) -> Optional[str]:
        """Generate refined, targeted questions for better retrieval (rlsapi pattern)"""
        
        # Use custom prompt if provided, otherwise use default template
        prompt_template = self.refinement_prompt if self.refinement_prompt else REFINEMENT_PROMPT_TEMPLATE
        prompt = prompt_template.format(question=query)
        
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
        assert self.embedding_provider is not None, "Engine must be started first"
        
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
        self, query_embedding: List[float], config: RAGConfig, query_text: str
    ) -> List[Dict[str, Any]]:
        """Retrieve similar documents from the database using hybrid search"""
        assert self.db_manager is not None, "Engine must be started first"
        assert config.model_name is not None, "Model name must be set"
        
        try:
            # Always use hybrid search (opinionated choice)
            assert config.max_chunks is not None and config.similarity_threshold is not None  # Set by start()
            hybrid_chunks = await self.db_manager.search_hybrid(
                query_embedding=query_embedding,
                query_text=query_text,
                model_name=config.model_name,
                limit=config.max_chunks * 2,  # Get extra for post-processing
                similarity_threshold=config.similarity_threshold,
            )

            # Convert to standard format
            documents = []
            for chunk in hybrid_chunks:
                # Use RRF score as the primary score
                score = chunk.get("rrf_score", 0.0)
                
                documents.append({
                    "text": chunk["text"],
                    "similarity_score": score,
                    "document_path": chunk.get("document_path", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "metadata": chunk.get("metadata", {}),
                    "vector_similarity": chunk.get("similarity"),
                    "bm25_rank": chunk.get("bm25_rank"),
                    "rrf_score": score,
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve similar documents: {e}")
            raise

    async def _rerank_documents(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents using a cross-encoder model for improved accuracy."""
        reranker = get_reranker()
        
        # Rerank all retrieved documents (no top_k limit)
        reranked = reranker.rerank(query, documents)
        
        # Use rerank_score as the new similarity_score for post-processing
        for doc in reranked:
            doc["similarity_score"] = doc["rerank_score"]
        
        logger.info(f"Reranked {len(reranked)} documents")
        return reranked

    def _post_process_results(
        self, documents: List[Dict[str, Any]], config: RAGConfig
    ) -> List[Dict[str, Any]]:
        """Post-process and filter results based on similarity and token limits"""
        # For hybrid search (RRF scores), skip similarity filtering since DB already filtered
        # RRF scores are small values (~0.01-0.05) and can't be compared to similarity thresholds
        is_hybrid = documents and "rrf_score" in documents[0]
        
        if is_hybrid:
            # Hybrid results are already filtered by DB and sorted by RRF score
            filtered = documents
        else:
            # Pure vector search: filter by similarity threshold
            filtered = [
                doc
                for doc in documents
                if doc["similarity_score"] >= config.similarity_threshold
            ]

        # Sort by similarity score (descending) - works for both RRF and cosine similarity
        filtered.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Limit by max_chunks
        filtered = filtered[: config.max_chunks]

        # Estimate token usage and truncate if needed
        total_tokens = 0
        final_docs = []

        assert config.max_tokens_in_context is not None  # Set by start()
        for doc in filtered:
            # Rough token estimation (1 token ≈ 4 characters)
            doc_tokens = len(doc["text"]) // 4

            if total_tokens + doc_tokens <= config.max_tokens_in_context:
                final_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break

        return final_docs

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM client (if available)"""
        if not self.llm_client:
            raise ValueError("No LLM client configured")

        # Use the LLM client's acomplete method
        return await self.llm_client.acomplete(prompt)

    def _get_features_used(
        self, config: RAGConfig, refined_questions: Optional[str]
    ) -> List[str]:
        """Get list of features used in this search"""
        features = [
            f"{config.model_name}_embeddings",
            "postgresql_vector_search",
            "hybrid_search",
            "similarity_post_processing",
        ]

        if config.enable_reranking:
            features.append("reranking")

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
    query: str, model_name: Optional[str] = None, **options
) -> RAGResult:
    """
    Convenience function for simple document search.

    Args:
        query: Search query
        model_name: Embedding model to use (auto-detected from database if None)
        **options: Additional search options

    Returns:
        RAGResult with documents and metadata
    """
    config = RAGConfig(
        model_name=model_name,
        **{k: v for k, v in options.items() if hasattr(RAGConfig, k)},
    )
    
    engine = UniversalRAGEngine(config)
    await engine.start()
    return await engine.search_documents(query, **options)


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
