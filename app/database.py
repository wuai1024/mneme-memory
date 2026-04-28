"""
Database module — SQLite with FTS5 for full-text search,
numpy-optimized cosine similarity for vector search.
Connection pooling for better performance.
"""
import os
import json
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional

import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")

# Connection pool with thread-local storage
_local = threading.local()


def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    """Get a database connection from the pool (thread-local)."""
    if not hasattr(_local, 'connections'):
        _local.connections = {}
    
    if path not in _local.connections:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.connections[path] = conn
    
    return _local.connections[path]


@contextmanager
def get_connection(path: str = DB_PATH):
    """Context manager for database connections."""
    conn = get_db(path)
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        pass  # Connection stays in pool


def close_all_connections():
    """Close all connections in the current thread."""
    if hasattr(_local, 'connections'):
        for conn in _local.connections.values():
            try:
                conn.close()
            except Exception:
                pass
        _local.connections.clear()


def init_db(path: str = DB_PATH):
    """Initialize database tables."""
    conn = get_db(path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id          TEXT PRIMARY KEY,
            subject     TEXT NOT NULL,
            predicate   TEXT NOT NULL,
            object      TEXT NOT NULL,
            embedding   TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id               TEXT PRIMARY KEY,
            conversation_id  TEXT NOT NULL,
            content          TEXT NOT NULL,
            embedding        TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            metadata         TEXT NOT NULL DEFAULT '{}'
        )
    """)

    # FTS5 virtual tables for keyword search
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(id, text)
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(id, text)
    """)

    # Create indexes for better query performance
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_created_at ON facts(created_at)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_summaries_conversation_id ON summaries(conversation_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at)
    """)

    conn.commit()


def generate_id(nbytes: int = 8) -> str:
    """Generate a random hex ID."""
    return secrets.token_hex(nbytes)


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Optimized cosine similarity using numpy."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def batch_cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Batch cosine similarity calculation."""
    # Normalize vectors
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    
    # Calculate similarities
    similarities = np.dot(vectors_norm, query_norm)
    return similarities


def search_vectors(
    query_vec: list[float],
    table: str,
    id_col: str,
    emb_col: str,
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[dict]:
    """
    Optimized vector search using numpy.
    Returns top-k results with score and id.
    """
    if table not in ("facts", "summaries"):
        raise ValueError(f"Unknown table: {table}")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(f"SELECT {id_col}, {emb_col} FROM {table}")
    rows = cur.fetchall()

    if not rows:
        return []

    # Prepare data for batch processing
    ids = []
    vectors = []
    for row in rows:
        ids.append(row[id_col])
        vectors.append(json.loads(row[emb_col]))

    # Convert to numpy arrays
    vectors_np = np.array(vectors, dtype=np.float32)
    query_np = np.array(query_vec, dtype=np.float32)

    # Batch similarity calculation
    similarities = batch_cosine_similarity(query_np, vectors_np)

    # Filter and sort
    results = []
    for i, (id_, score) in enumerate(zip(ids, similarities)):
        if score >= threshold:
            results.append({"id": id_, "score": float(score)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_vectors_batch(
    query_vecs: list[list[float]],
    table: str,
    id_col: str,
    emb_col: str,
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[list[dict]]:
    """
    Batch vector search for multiple queries.
    Returns list of results for each query.
    """
    if table not in ("facts", "summaries"):
        raise ValueError(f"Unknown table: {table}")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(f"SELECT {id_col}, {emb_col} FROM {table}")
    rows = cur.fetchall()

    if not rows:
        return [[] for _ in query_vecs]

    # Prepare data
    ids = []
    vectors = []
    for row in rows:
        ids.append(row[id_col])
        vectors.append(json.loads(row[emb_col]))

    vectors_np = np.array(vectors, dtype=np.float32)
    query_np = np.array(query_vecs, dtype=np.float32)

    # Normalize vectors once
    vectors_norm = vectors_np / (np.linalg.norm(vectors_np, axis=1, keepdims=True) + 1e-10)
    query_norm = query_np / (np.linalg.norm(query_np, axis=1, keepdims=True) + 1e-10)

    # Batch similarity: (n_queries, n_vectors)
    similarity_matrix = np.dot(query_norm, vectors_norm.T)

    # Process each query
    all_results = []
    for i in range(len(query_vecs)):
        results = []
        for j, (id_, score) in enumerate(zip(ids, similarity_matrix[i])):
            if score >= threshold:
                results.append({"id": id_, "score": float(score)})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        all_results.append(results[:top_k])

    return all_results


def get_table_stats() -> dict:
    """Get statistics about the database tables."""
    conn = get_db()
    cur = conn.cursor()
    
    stats = {}
    
    cur.execute("SELECT COUNT(*) FROM facts")
    stats["facts_count"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM summaries")
    stats["summaries_count"] = cur.fetchone()[0]
    
    cur.execute("SELECT MIN(created_at) FROM facts")
    stats["oldest_fact"] = cur.fetchone()[0]
    
    cur.execute("SELECT MAX(created_at) FROM facts")
    stats["newest_fact"] = cur.fetchone()[0]
    
    return stats


def vacuum_database():
    """Vacuum the database to reclaim space."""
    conn = get_db()
    conn.execute("VACUUM")
