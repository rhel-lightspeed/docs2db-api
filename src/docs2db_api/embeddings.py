"""Embedding models and generation logic for query lookup."""

from typing import Any, Dict, List

import structlog
import torch
from transformers import AutoModel, AutoTokenizer

from docs2db_api.config import settings

logger = structlog.get_logger(__name__)


def get_optimal_device() -> str:
    """Detect and return the optimal device for embedding generation."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_to_device(model_or_tensor, device: str):
    """Move a model or tensor to the specified device."""
    try:
        return model_or_tensor.to(device)
    except Exception:
        logger.error(f"Failed to move model to device: {device}")
        return model_or_tensor


class EmbeddingProvider:
    """Base class for embedding providers."""

    def __init__(self, model_name: str, config: Dict[str, Any], device: str):
        self.model_name = model_name
        self.config = config
        self.model = model_name  # Full model identifier (e.g., "ibm-granite/granite-embedding-30m-english")
        self.dimensions = config["dimensions"]
        self.device = device

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError


class GraniteEmbeddingProvider(EmbeddingProvider):
    """Granite embedding provider using CLS token pooling."""

    def __init__(self, model_name: str, config: Dict[str, Any], device: str):
        super().__init__(model_name, config, device)
        self._model = None
        self._tokenizer = None

    def _get_model_and_tokenizer(self):
        """Get or create the Granite model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            # Set MPS memory limits to prevent memory leaks
            if self.device == "mps":
                torch.mps.set_per_process_memory_fraction(
                    0.4
                )  # Limit to 40% of memory per worker

            try:
                offline = settings.embedding.offline
                self._model = AutoModel.from_pretrained(
                    self.model, local_files_only=offline
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model, local_files_only=offline
                )
            except Exception as e:
                if settings.embedding.offline:
                    raise ValueError(
                        f"Granite model '{self.model}' not found locally (DOCS2DB_OFFLINE=true). "
                        f"Download it first by running without DOCS2DB_OFFLINE set. "
                        f"Original error: {e}"
                    ) from e
                raise ValueError(
                    f"Failed to load Granite model '{self.model}'. "
                    f"Check your internet connection for first-time download. "
                    f"Original error: {e}"
                ) from e

            self._model.eval()
            self._model = move_to_device(self._model, self.device)

        return self._model, self._tokenizer

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Granite model with CLS pooling."""
        model, tokenizer = self._get_model_and_tokenizer()

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move input tensors to the same device as the model
        inputs = {k: move_to_device(v, self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use CLS token pooling (first token)
        # Granite uses CLS pooling as mentioned in the documentation
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Normalize embeddings for better similarity scores
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        # Convert to list
        result = embeddings.cpu().numpy().tolist()

        return result


EMBEDDING_CONFIGS = {
    "ibm-granite/granite-embedding-30m-english": {
        "keyword": "gran",
        "dimensions": 384,
        "provider": "granite",
        "cls": GraniteEmbeddingProvider,
    },
}
