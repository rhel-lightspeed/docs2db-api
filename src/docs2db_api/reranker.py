"""Reranker implementation for improving search result quality."""

from typing import Any, Dict, List, Optional

import structlog
from sentence_transformers import CrossEncoder

logger = structlog.get_logger(__name__)


class Reranker:
    """
    Reranker that uses a cross-encoder model to reorder search results.
    
    Cross-encoders jointly encode query and document, providing more accurate
    relevance scores than bi-encoders (embedding models) which encode separately.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for the cross-encoder.
                       Default is a fast, accurate model trained on MS MARCO.
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model when first needed."""
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of document dictionaries containing 'text' and other fields
            top_k: Number of top results to return (None = return all)

        Returns:
            List of reranked documents with added 'rerank_score' field
        """
        if not documents:
            return documents

        # Prepare query-document pairs for the cross-encoder
        pairs = [[query, doc["text"]] for doc in documents]

        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs)

        # Add rerank scores to documents
        for doc, score in zip(documents, scores, strict=False):
            doc["rerank_score"] = float(score)

        # Sort by rerank score (descending)
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        # Limit to top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"Reranked {len(documents)} documents")
        return reranked


# Default reranker instance (singleton pattern)
_default_reranker = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    """
    Get or create the default reranker instance.

    Args:
        model_name: HuggingFace model name for the cross-encoder

    Returns:
        Reranker instance
    """
    global _default_reranker
    if _default_reranker is None or _default_reranker.model_name != model_name:
        _default_reranker = Reranker(model_name)
    return _default_reranker

