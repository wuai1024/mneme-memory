"""
Embedding module — wraps sentence-transformers for consistent interface.
Supports custom cache folder via MODEL_CACHE env var.
"""
import os
import threading
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_CACHE = os.environ.get("MODEL_CACHE", None)

# Thread-safe singleton model loader
_model = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                kwargs = {"cache_folder": MODEL_CACHE} if MODEL_CACHE else {}
                _model = SentenceTransformer(MODEL_NAME, **kwargs)
    return _model


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    return _get_model().get_sentence_embedding_dimension()


def embed(texts: list[str]) -> list[list[float]]:
    """
    Returns list of embedding vectors, one per input text.
    Each vector is a list of floats (not a numpy array — JSON serializable).
    """
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vectors]
