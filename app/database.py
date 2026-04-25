"""
Database module — SQLite with FTS5 for full-text search,
pure-Python cosine similarity for vector search.
"""
import os
import json
import math
import secrets
import sqlite3
from contextlib import contextmanager
from typing import Optional

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")


def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection():
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: str = DB_PATH):
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

    conn.commit()
    conn.close()


def generate_id(nbytes: int = 8) -> str:
    return secrets.token_hex(nbytes)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_vectors(
    query_vec: list[float],
    table: str,
    id_col: str,
    emb_col: str,
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[dict]:
    """
    Pure-Python cosine similarity search over stored vectors.
    `table` must be 'facts' or 'summaries'.
    Returns top-k results with score and id.
    """
    if table not in ("facts", "summaries"):
        raise ValueError(f"Unknown table: {table}")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(f"SELECT {id_col}, {emb_col} FROM {table}")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for row in rows:
        stored_vec = json.loads(row[emb_col])
        score = cosine_similarity(query_vec, stored_vec)
        if score >= threshold:
            scored.append({"id": row[id_col], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
