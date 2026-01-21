"""Configuration settings for docs2db-api using Pydantic."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM configuration for query refinement."""

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for OpenAI-compatible API (e.g., Ollama)",
    )
    model: str = Field(
        default="qwen2.5:7b-instruct",
        description="Model name to use for query refinement",
    )
    timeout: float = Field(default=30.0, description="HTTP client timeout in seconds")
    temperature: float = Field(default=0.7, description="LLM temperature for generation")
    max_tokens: int = Field(default=500, description="Maximum tokens for LLM response")

    model_config = SettingsConfigDict(
        env_prefix="DOCS2DB_LLM_", env_file=".env", extra="ignore"
    )


class DatabaseSettings(BaseSettings):
    """Database connection configuration.

    Environment variables (with DOCS2DB_DB_ prefix):
    - DOCS2DB_DB_HOST: PostgreSQL host
    - DOCS2DB_DB_PORT: PostgreSQL port
    - DOCS2DB_DB_DATABASE: PostgreSQL database name
    - DOCS2DB_DB_USER: PostgreSQL user
    - DOCS2DB_DB_PASSWORD: PostgreSQL password
    - DOCS2DB_DB_URL: PostgreSQL connection URL (alternative to individual settings)
    """

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(default="ragdb", description="PostgreSQL database name")
    user: str = Field(default="postgres", description="PostgreSQL user")
    password: str = Field(default="postgres", description="PostgreSQL password")
    url: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection URL (alternative to individual settings)",
    )

    model_config = SettingsConfigDict(
        env_prefix="DOCS2DB_DB_", env_file=".env", extra="ignore"
    )


class RAGSettings(BaseSettings):
    """RAG engine configuration.

    Environment variables (with DOCS2DB_RAG_ prefix):
    - DOCS2DB_RAG_SIMILARITY_THRESHOLD
    - DOCS2DB_RAG_MAX_CHUNKS
    - DOCS2DB_RAG_MAX_TOKENS_IN_CONTEXT
    - DOCS2DB_RAG_ENABLE_QUESTION_REFINEMENT
    - DOCS2DB_RAG_ENABLE_RERANKING
    - DOCS2DB_RAG_REFINEMENT_QUESTIONS_COUNT
    - DOCS2DB_RAG_REFINEMENT_PROMPT

    These are code-level defaults. The actual hierarchy is:
    CLI/kwargs → environment → database → these defaults
    """

    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity score for results"
    )
    max_chunks: int = Field(
        default=10, ge=1, description="Maximum number of chunks to retrieve"
    )
    max_tokens_in_context: int = Field(
        default=4096, ge=1, description="Maximum tokens to include in context"
    )
    enable_question_refinement: bool = Field(
        default=True, description="Enable LLM-based query refinement"
    )
    enable_reranking: bool = Field(
        default=True, description="Enable cross-encoder reranking"
    )
    refinement_questions_count: int = Field(
        default=5, ge=1, description="Number of refined questions to generate"
    )
    refinement_prompt: Optional[str] = Field(
        default=None, description="Custom prompt template for query refinement"
    )

    model_config = SettingsConfigDict(
        env_prefix="DOCS2DB_RAG_", env_file=".env", extra="ignore"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    model_config = SettingsConfigDict(
        env_prefix="DOCS2DB_", env_file=".env", extra="ignore"
    )


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    offline: bool = Field(
        default=False,
        description="Run in offline mode (only use locally cached models)",
    )

    model_config = SettingsConfigDict(
        env_prefix="DOCS2DB_", env_file=".env", extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Global settings instance
settings = Settings()

