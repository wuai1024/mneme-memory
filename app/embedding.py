"""
Embedding module — wraps sentence-transformers for consistent interface.
Supports custom cache folder via MODEL_CACHE env var.
Automatically uses GPU (CUDA) if available.
"""
import os
import torch
import threading
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_CACHE = os.environ.get("MODEL_CACHE", None)

# Thread-safe singleton model loader
_model = None
_model_lock = threading.Lock()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                kwargs = {"cache_folder": MODEL_CACHE} if MODEL_CACHE else {}
                _model = SentenceTransformer(MODEL_NAME, device=DEVICE, **kwargs)
    return _model


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    return _get_model().get_sentence_embedding_dimension()


def embed(texts: list[str]) -> list[list[float]]:
    """
    Returns list of embedding vectors, one per input text.
    Each vector is a list of floats (not a numpy array — JSON serializable).
    Uses GPU automatically if CUDA is available.
    """
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vectors]
