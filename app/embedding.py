"""
Embedding module — wraps sentence-transformers for consistent interface.
Supports custom cache folder via MODEL_CACHE env var.
CPU-only (no GPU dependency required).
"""
import os
import threading
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_CACHE = os.environ.get("MODEL_CACHE", None)

# Thread-safe singleton model loader
_model = None
_model_lock = threading.Lock()

DEVICE = "cpu"


def _get_model() -> SentenceTransformer:
    """Get or create the sentence transformer model (thread-safe singleton)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                kwargs = {"cache_folder": MODEL_CACHE} if MODEL_CACHE else {}
                _model = SentenceTransformer(MODEL_NAME, device=DEVICE, **kwargs)
    return _model


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Get the dimension of the embedding vectors."""
    return _get_model().get_sentence_embedding_dimension()


def embed(texts: list[str]) -> list[list[float]]:
    """
    Returns list of embedding vectors, one per input text.
    Each vector is a list of floats (not a numpy array — JSON serializable).
    """
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Batch embedding for better performance with large inputs.
    """
    model = _get_model()
    vectors = model.encode(
        texts, 
        normalize_embeddings=True, 
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False
    )
    return [v.tolist() for v in vectors]


def get_model_info() -> dict:
    """Get information about the current model."""
    model = _get_model()
    return {
        "model_name": MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "device": DEVICE,
        "max_seq_length": model.max_seq_length,
    }
