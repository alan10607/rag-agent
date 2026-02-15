"""
VectorSearcher - Embedding Generator

Wraps the sentence-transformers library to provide a simple interface
for converting text into vector embeddings.

Model: all-MiniLM-L6-v2 (384-dimensional vectors)
"""

from sentence_transformers import SentenceTransformer

from app import config
from app.logger import get_logger

logger = get_logger(__name__)

# Module-level model cache (loaded once on first use)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", config.EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model


def encode(text: str) -> list[float]:
    """Encode a single text string into a vector embedding.

    Args:
        text: The input text to encode.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def encode_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of text strings into vector embeddings.

    Args:
        texts: A list of input texts to encode.

    Returns:
        A list of embedding vectors (each a list of floats).
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    logger.info("Encoded %d texts into embeddings.", len(texts))
    return embeddings.tolist()
