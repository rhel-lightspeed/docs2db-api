#!/usr/bin/env python3
"""
Generic RAG REST API
====================

FastAPI-based REST API that provides universal access to the RAG engine.
This API is framework-agnostic and can be used by any HTTP client.

Features:
- Simple document search endpoint
- Search + response generation endpoint
- Model selection and configuration
- OpenAPI documentation
- Tool calling compatible responses

Usage:
    uvicorn codex.rag.api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from codex.embeddings import EMBEDDING_CONFIGS
from codex.rag.engine import RAGConfig, RAGResult, UniversalRAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Universal RAG API",
    description="Generic RAG API for document search and response generation using Codex database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SearchRequest(BaseModel):
    """Request model for document search"""

    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    model_name: Optional[str] = Field(
        "granite-30m-english", description="Embedding model to use"
    )
    similarity_threshold: Optional[float] = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    max_chunks: Optional[int] = Field(
        10, ge=1, le=50, description="Maximum number of document chunks to return"
    )
    max_tokens_in_context: Optional[int] = Field(
        4096, ge=512, le=16384, description="Maximum tokens in context"
    )
    enable_question_refinement: Optional[bool] = Field(
        True, description="Enable question refinement"
    )
    enable_hybrid_search: Optional[bool] = Field(
        True, description="Enable hybrid search"
    )


class DocumentResult(BaseModel):
    """Individual document result"""

    text: str = Field(..., description="Document text content")
    similarity_score: float = Field(..., description="Similarity score (0.0-1.0)")
    document_path: str = Field("", description="Source document path")
    chunk_index: int = Field(0, description="Chunk index within document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResponse(BaseModel):
    """Response model for document search"""

    query: str = Field(..., description="Original search query")
    documents: List[DocumentResult] = Field(..., description="Retrieved documents")
    refined_questions: Optional[str] = Field(
        None, description="Refined questions (if enabled)"
    )
    metadata: Dict[str, Any] = Field(..., description="Search metadata and statistics")


class GenerateRequest(BaseModel):
    """Request model for search + generation"""

    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    model_name: Optional[str] = Field(
        "granite-30m-english", description="Embedding model to use"
    )
    similarity_threshold: Optional[float] = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    max_chunks: Optional[int] = Field(
        10, ge=1, le=50, description="Maximum number of document chunks to use"
    )
    max_tokens_in_context: Optional[int] = Field(
        4096, ge=512, le=16384, description="Maximum tokens in context"
    )
    enable_question_refinement: Optional[bool] = Field(
        True, description="Enable question refinement"
    )
    enable_hybrid_search: Optional[bool] = Field(
        True, description="Enable hybrid search"
    )


class GenerateResponse(BaseModel):
    """Response model for search + generation"""

    query: str = Field(..., description="Original search query")
    response: str = Field(..., description="Generated response")
    documents: List[DocumentResult] = Field(
        ..., description="Retrieved documents used for generation"
    )
    refined_questions: Optional[str] = Field(
        None, description="Refined questions (if enabled)"
    )
    metadata: Dict[str, Any] = Field(..., description="Search metadata and statistics")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    models_available: List[str] = Field(..., description="Available embedding models")
    database_status: str = Field(..., description="Database connection status")


class ModelsResponse(BaseModel):
    """Available models response"""

    models: Dict[str, Dict[str, Any]] = Field(
        ..., description="Available embedding models and their configurations"
    )


# Global RAG engine instance (will be initialized on startup)
rag_engine: Optional[UniversalRAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    try:
        config = RAGConfig()  # Default configuration
        rag_engine = UniversalRAGEngine(config)
        logger.info("✅ RAG API initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global rag_engine
    if rag_engine:
        await rag_engine.close()
        logger.info("✅ RAG API shutdown complete")


# API Endpoints


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        if rag_engine and rag_engine.db_manager:
            await rag_engine.db_manager.get_connection_pool()
            db_status = "connected"
        else:
            db_status = "disconnected"

        return HealthResponse(
            status="healthy",
            models_available=list(EMBEDDING_CONFIGS.keys()),
            database_status=db_status,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """Get available embedding models"""
    # Create serializable version without class objects
    serializable_configs = {}
    for name, config in EMBEDDING_CONFIGS.items():
        serializable_config = {k: v for k, v in config.items() if k != "cls"}
        # Add provider class name as string instead of class object
        if "cls" in config:
            serializable_config["provider_class"] = config["cls"].__name__
        serializable_configs[name] = serializable_config

    return ModelsResponse(models=serializable_configs)


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for relevant documents using RAG.

    This endpoint performs semantic search using embeddings and returns
    relevant document chunks with similarity scores. Perfect for tool calling
    and integration with AI frameworks.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        logger.info(
            f"🔍 Search request: {request.query[:100]}... (model: {request.model_name})"
        )

        # Convert request to search options
        search_options = {
            "model_name": request.model_name,
            "similarity_threshold": request.similarity_threshold,
            "max_chunks": request.max_chunks,
            "max_tokens_in_context": request.max_tokens_in_context,
            "enable_question_refinement": request.enable_question_refinement,
            "enable_hybrid_search": request.enable_hybrid_search,
        }

        # Perform search
        result = await rag_engine.search_documents(request.query, **search_options)

        # Convert to response format
        documents = [
            DocumentResult(
                text=doc["text"],
                similarity_score=doc["similarity_score"],
                document_path=doc["document_path"],
                chunk_index=doc["chunk_index"],
                metadata=doc["metadata"],
            )
            for doc in result.documents
        ]

        logger.info(f"✅ Search completed: {len(documents)} documents returned")

        return SearchResponse(
            query=result.query,
            documents=documents,
            refined_questions=result.refined_questions,
            metadata=result.metadata or {},
        )

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search operation failed")


@app.post("/generate", response_model=GenerateResponse)
async def search_and_generate(request: GenerateRequest):
    """
    Search for documents and generate a response using LLM.

    This endpoint combines document retrieval with response generation
    for complete RAG functionality. Requires LLM integration.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        logger.info(
            f"🚀 Generate request: {request.query[:100]}... (model: {request.model_name})"
        )

        # Convert request to search options
        search_options = {
            "model_name": request.model_name,
            "similarity_threshold": request.similarity_threshold,
            "max_chunks": request.max_chunks,
            "max_tokens_in_context": request.max_tokens_in_context,
            "enable_question_refinement": request.enable_question_refinement,
            "enable_hybrid_search": request.enable_hybrid_search,
        }

        # Perform search and generation
        result = await rag_engine.search_and_generate(request.query, **search_options)

        # Convert to response format
        documents = [
            DocumentResult(
                text=doc["text"],
                similarity_score=doc["similarity_score"],
                document_path=doc["document_path"],
                chunk_index=doc["chunk_index"],
                metadata=doc["metadata"],
            )
            for doc in result.documents
        ]

        logger.info(
            f"✅ Generation completed: {len(documents)} documents, response generated"
        )

        return GenerateResponse(
            query=result.query,
            response=result.response or "Response generation not available",
            documents=documents,
            refined_questions=result.refined_questions,
            metadata=result.metadata or {},
        )

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation operation failed")


# Convenience endpoints with query parameters (for simple integrations)


@app.get("/search")
async def search_documents_get(
    q: str = Query(..., description="Search query", min_length=1, max_length=1000),
    model: str = Query("granite-30m-english", description="Embedding model"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    refine: bool = Query(True, description="Enable question refinement"),
):
    """
    Simple GET endpoint for document search.
    Useful for quick testing and simple integrations.
    """
    request = SearchRequest(
        query=q,
        model_name=model,
        similarity_threshold=threshold,
        max_chunks=limit,
        enable_question_refinement=refine,
        max_tokens_in_context=4096,
        enable_hybrid_search=True,
    )
    return await search_documents(request)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
