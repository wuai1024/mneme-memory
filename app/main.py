"""
Mneme Memory Service — FastAPI entry point.
All write/read endpoints require X-API-Key authentication.
"""
import os
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    FactCreate, FactResponse,
    SummaryCreate, SummaryResponse,
    SearchRequest, SearchResponse, HealthResponse
)
from .database import init_db, get_db, generate_id, search_vectors
from .embedding import embed, get_embedding_dimension

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.environ.get("MEMORY_API_KEY", "")
SKIP_AUTH = os.environ.get("SKIP_AUTH", "").lower() in ("1", "true", "yes")


async def verify_api_key(request: Request):
    """Fail if SKIP_AUTH is false and X-API-Key header is missing or wrong."""
    if SKIP_AUTH or not API_KEY:
        return
    provided = request.headers.get("X-API-Key", "")
    if not provided or provided != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"[{datetime.now().isoformat()}] Mneme memory service started")
    yield
    print(f"[{datetime.now().isoformat()}] Mneme memory service stopped.")


app = FastAPI(
    title="Mneme Memory Service",
    version="1.0.0",
    lifespan=lifespan,
    description="Semantic long-term memory for AI agents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        embedding_model=MODEL_NAME,
        embedding_dimension=get_embedding_dimension()
    )

# ── Facts ───────────────────────────────────────────────────────────────────

@app.post("/facts", response_model=FactResponse, status_code=201, dependencies=[Depends(verify_api_key)])
def create_fact(fact: FactCreate):
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    fid = generate_id()

    text = f"{fact.subject} {fact.predicate} {fact.object}"
    vectors = embed([text])
    embedding = json.dumps(vectors[0])

    cur.execute(
        "INSERT INTO facts (id,subject,predicate,object,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?,?)",
        (fid, fact.subject, fact.predicate, fact.object, embedding, now, now, json.dumps(fact.metadata or {}))
    )
    cur.execute("INSERT INTO facts_fts (id,text) VALUES (?,?)", (fid, text))
    conn.commit()
    conn.close()

    return FactResponse(
        id=fid, subject=fact.subject, predicate=fact.predicate, object=fact.object,
        created_at=now, updated_at=now, metadata=fact.metadata or {}
    )


@app.get("/facts", response_model=list[FactResponse], dependencies=[Depends(verify_api_key)])
def list_facts(limit: int = Query(default=100, ge=1, le=1000)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM facts ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        FactResponse(
            id=r["id"], subject=r["subject"], predicate=r["predicate"], object=r["object"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            metadata=json.loads(r["metadata"] or "{}")
        ) for r in rows
    ]


@app.get("/facts/{fact_id}", response_model=FactResponse, dependencies=[Depends(verify_api_key)])
def get_fact(fact_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM facts WHERE id=?", (fact_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        raise HTTPException(status_code=404, detail="Fact not found")
    return FactResponse(
        id=r["id"], subject=r["subject"], predicate=r["predicate"], object=r["object"],
        created_at=r["created_at"], updated_at=r["updated_at"],
        metadata=json.loads(r["metadata"] or "{}")
    )


@app.delete("/facts/{fact_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_fact(fact_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM facts WHERE id=?", (fact_id,))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Fact not found")
    try:
        cur.execute("DELETE FROM facts_fts WHERE id=?", (fact_id,))
    except Exception:
        pass
    conn.commit()
    conn.close()

# ── Summaries ───────────────────────────────────────────────────────────────

@app.post("/summaries", response_model=SummaryResponse, status_code=201, dependencies=[Depends(verify_api_key)])
def create_summary(summary: SummaryCreate):
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    sid = generate_id()

    vectors = embed([summary.content])
    embedding = json.dumps(vectors[0])

    cur.execute(
        "INSERT INTO summaries (id,conversation_id,content,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?)",
        (sid, summary.conversation_id, summary.content, embedding, now, now, json.dumps(summary.metadata or {}))
    )
    cur.execute("INSERT INTO summaries_fts (id,text) VALUES (?,?)", (sid, summary.content))
    conn.commit()
    conn.close()

    return SummaryResponse(
        id=sid, conversation_id=summary.conversation_id, content=summary.content,
        created_at=now, updated_at=now, metadata=summary.metadata or {}
    )


@app.get("/summaries", response_model=list[SummaryResponse], dependencies=[Depends(verify_api_key)])
def list_summaries(
    conversation_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000)
):
    conn = get_db()
    cur = conn.cursor()
    if conversation_id:
        cur.execute(
            "SELECT * FROM summaries WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
            (conversation_id, limit)
        )
    else:
        cur.execute("SELECT * FROM summaries ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        SummaryResponse(
            id=r["id"], conversation_id=r["conversation_id"], content=r["content"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            metadata=json.loads(r["metadata"] or "{}")
        ) for r in rows
    ]


@app.delete("/summaries/{summary_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_summary(summary_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM summaries WHERE id=?", (summary_id,))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Summary not found")
    try:
        cur.execute("DELETE FROM summaries_fts WHERE id=?", (summary_id,))
    except Exception:
        pass
    conn.commit()
    conn.close()

# ── Search ──────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search(req: SearchRequest):
    query_vec = embed([req.query])[0]

    fact_results = search_vectors(query_vec, "facts", "id", "embedding", req.top_k)
    summary_results = search_vectors(query_vec, "summaries", "id", "embedding", req.top_k)

    conn = get_db()
    cur = conn.cursor()

    facts = []
    for item in fact_results:
        cur.execute("SELECT * FROM facts WHERE id=?", (item["id"],))
        r = cur.fetchone()
        if r:
            facts.append(FactResponse(
                id=r["id"], subject=r["subject"], predicate=r["predicate"], object=r["object"],
                created_at=r["created_at"], updated_at=r["updated_at"],
                metadata=json.loads(r["metadata"] or "{}")
            ))

    summaries = []
    for item in summary_results:
        cur.execute("SELECT * FROM summaries WHERE id=?", (item["id"],))
        r = cur.fetchone()
        if r:
            summaries.append(SummaryResponse(
                id=r["id"], conversation_id=r["conversation_id"], content=r["content"],
                created_at=r["created_at"], updated_at=r["updated_at"],
                metadata=json.loads(r["metadata"] or "{}")
            ))

    conn.close()
    return SearchResponse(facts=facts, summaries=summaries)
