"""
Mneme Memory Service — FastAPI entry point.
All write/read endpoints require X-API-Key authentication.
"""
import os
import json
import hmac
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    FactCreate, FactUpdate, FactResponse,
    SummaryCreate, SummaryResponse,
    SearchRequest, SearchResponse, HealthResponse,
    BatchFactCreate, BatchFactDelete, BatchSummaryCreate,
    ExportResponse, ImportRequest, ImportResponse,
    MetricsResponse
)
from .database import (
    init_db, get_db, generate_id, search_vectors, 
    search_vectors_batch, get_table_stats, vacuum_database,
    close_all_connections
)
from .embedding import embed, embed_batch, get_model_info

# Configuration
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.environ.get("MEMORY_API_KEY", "")
SKIP_AUTH = os.environ.get("SKIP_AUTH", "").lower() in ("1", "true", "yes")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mneme")

# Metrics storage
metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "facts_created": 0,
    "facts_deleted": 0,
    "summaries_created": 0,
    "summaries_deleted": 0,
    "searches_performed": 0,
    "start_time": time.time()
}


async def verify_api_key(request: Request):
    """Fail if SKIP_AUTH is false and X-API-Key header is missing or wrong."""
    if SKIP_AUTH or not API_KEY:
        return
    provided = request.headers.get("X-API-Key", "")
    if not provided or not hmac.compare_digest(provided, API_KEY):
        logger.warning(f"Unauthorized access attempt from {request.client.host}")
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    init_db()
    logger.info(f"Mneme memory service started at {datetime.now().isoformat()}")
    yield
    close_all_connections()
    logger.info(f"Mneme memory service stopped at {datetime.now().isoformat()}")


app = FastAPI(
    title="Mneme Memory Service",
    version="1.1.0",
    lifespan=lifespan,
    description="Semantic long-term memory for AI agents.",
)

# CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics."""
    metrics["requests_total"] += 1
    start_time = time.time()
    
    try:
        response = await call_next(request)
        if response.status_code < 400:
            metrics["requests_success"] += 1
        else:
            metrics["requests_error"] += 1
        return response
    except Exception as e:
        metrics["requests_error"] += 1
        raise e
    finally:
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {duration:.3f}s")


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    model_info = get_model_info()
    return HealthResponse(
        status="ok",
        embedding_model=MODEL_NAME,
        embedding_dimension=model_info["embedding_dimension"],
        version="1.1.0"
    )


# ── Facts ───────────────────────────────────────────────────────────────────

@app.post("/facts", response_model=FactResponse, status_code=201, dependencies=[Depends(verify_api_key)])
def create_fact(fact: FactCreate):
    """Create a new fact."""
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    fid = generate_id()

    text = f"{fact.subject} {fact.predicate} {fact.object}"
    vectors = embed([text])
    embedding = json.dumps(vectors[0])

    try:
        cur.execute(
            "INSERT INTO facts (id,subject,predicate,object,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?,?)",
            (fid, fact.subject, fact.predicate, fact.object, embedding, now, now, json.dumps(fact.metadata or {}))
        )
        cur.execute("INSERT INTO facts_fts (id,text) VALUES (?,?)", (fid, text))
        conn.commit()
        metrics["facts_created"] += 1
        logger.info(f"Created fact: {fid}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create fact: {e}")
        raise HTTPException(status_code=500, detail="Failed to create fact")

    return FactResponse(
        id=fid, subject=fact.subject, predicate=fact.predicate, object=fact.object,
        created_at=now, updated_at=now, metadata=fact.metadata or {}
    )


@app.post("/facts/batch", response_model=list[FactResponse], status_code=201, dependencies=[Depends(verify_api_key)])
def create_facts_batch(batch: BatchFactCreate):
    """Create multiple facts in batch."""
    if not batch.facts:
        raise HTTPException(status_code=400, detail="No facts provided")
    
    if len(batch.facts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 facts per batch")

    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    
    # Prepare texts for batch embedding
    texts = [f"{f.subject} {f.predicate} {f.object}" for f in batch.facts]
    vectors = embed_batch(texts)
    
    results = []
    try:
        for i, (fact, vector) in enumerate(zip(batch.facts, vectors)):
            fid = generate_id()
            embedding = json.dumps(vector)
            
            cur.execute(
                "INSERT INTO facts (id,subject,predicate,object,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?,?)",
                (fid, fact.subject, fact.predicate, fact.object, embedding, now, now, json.dumps(fact.metadata or {}))
            )
            cur.execute("INSERT INTO facts_fts (id,text) VALUES (?,?)", (fid, texts[i]))
            
            results.append(FactResponse(
                id=fid, subject=fact.subject, predicate=fact.predicate, object=fact.object,
                created_at=now, updated_at=now, metadata=fact.metadata or {}
            ))
        
        conn.commit()
        metrics["facts_created"] += len(results)
        logger.info(f"Created {len(results)} facts in batch")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create facts batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to create facts batch")

    return results


@app.get("/facts", response_model=list[FactResponse], dependencies=[Depends(verify_api_key)])
def list_facts(limit: int = Query(default=100, ge=1, le=1000)):
    """List facts with pagination."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM facts ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [
        FactResponse(
            id=r["id"], subject=r["subject"], predicate=r["predicate"], object=r["object"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            metadata=json.loads(r["metadata"] or "{}")
        ) for r in rows
    ]


@app.get("/facts/{fact_id}", response_model=FactResponse, dependencies=[Depends(verify_api_key)])
def get_fact(fact_id: str):
    """Get a single fact by ID."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM facts WHERE id=?", (fact_id,))
    r = cur.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Fact not found")
    return FactResponse(
        id=r["id"], subject=r["subject"], predicate=r["predicate"], object=r["object"],
        created_at=r["created_at"], updated_at=r["updated_at"],
        metadata=json.loads(r["metadata"] or "{}")
    )


@app.put("/facts/{fact_id}", response_model=FactResponse, dependencies=[Depends(verify_api_key)])
def update_fact(fact_id: str, fact: FactUpdate):
    """Update an existing fact."""
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    
    # Check if fact exists
    cur.execute("SELECT * FROM facts WHERE id=?", (fact_id,))
    existing = cur.fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Fact not found")
    
    # Prepare update fields
    subject = fact.subject if fact.subject is not None else existing["subject"]
    predicate = fact.predicate if fact.predicate is not None else existing["predicate"]
    object_ = fact.object if fact.object is not None else existing["object"]
    metadata = fact.metadata if fact.metadata is not None else json.loads(existing["metadata"] or "{}")
    
    # Recalculate embedding if content changed
    text = f"{subject} {predicate} {object_}"
    vectors = embed([text])
    embedding = json.dumps(vectors[0])
    
    try:
        cur.execute(
            "UPDATE facts SET subject=?, predicate=?, object=?, embedding=?, updated_at=?, metadata=? WHERE id=?",
            (subject, predicate, object_, embedding, now, json.dumps(metadata), fact_id)
        )
        # Update FTS
        cur.execute("DELETE FROM facts_fts WHERE id=?", (fact_id,))
        cur.execute("INSERT INTO facts_fts (id,text) VALUES (?,?)", (fact_id, text))
        conn.commit()
        logger.info(f"Updated fact: {fact_id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update fact: {e}")
        raise HTTPException(status_code=500, detail="Failed to update fact")
    
    return FactResponse(
        id=fact_id, subject=subject, predicate=predicate, object=object_,
        created_at=existing["created_at"], updated_at=now, metadata=metadata
    )


@app.delete("/facts/{fact_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_fact(fact_id: str):
    """Delete a fact."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM facts WHERE id=?", (fact_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Fact not found")
    try:
        cur.execute("DELETE FROM facts_fts WHERE id=?", (fact_id,))
    except Exception:
        pass
    conn.commit()
    metrics["facts_deleted"] += 1
    logger.info(f"Deleted fact: {fact_id}")


@app.delete("/facts/batch", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_facts_batch(batch: BatchFactDelete):
    """Delete multiple facts in batch."""
    if not batch.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    
    if len(batch.ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 IDs per batch")

    conn = get_db()
    cur = conn.cursor()
    
    try:
        placeholders = ",".join(["?" for _ in batch.ids])
        cur.execute(f"DELETE FROM facts WHERE id IN ({placeholders})", batch.ids)
        deleted_count = cur.rowcount
        
        # Clean up FTS
        for fact_id in batch.ids:
            try:
                cur.execute("DELETE FROM facts_fts WHERE id=?", (fact_id,))
            except Exception:
                pass
        
        conn.commit()
        metrics["facts_deleted"] += deleted_count
        logger.info(f"Deleted {deleted_count} facts in batch")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to delete facts batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete facts batch")


# ── Summaries ───────────────────────────────────────────────────────────────

@app.post("/summaries", response_model=SummaryResponse, status_code=201, dependencies=[Depends(verify_api_key)])
def create_summary(summary: SummaryCreate):
    """Create a new summary."""
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    sid = generate_id()

    vectors = embed([summary.content])
    embedding = json.dumps(vectors[0])

    try:
        cur.execute(
            "INSERT INTO summaries (id,conversation_id,content,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?)",
            (sid, summary.conversation_id, summary.content, embedding, now, now, json.dumps(summary.metadata or {}))
        )
        cur.execute("INSERT INTO summaries_fts (id,text) VALUES (?,?)", (sid, summary.content))
        conn.commit()
        metrics["summaries_created"] += 1
        logger.info(f"Created summary: {sid}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to create summary")

    return SummaryResponse(
        id=sid, conversation_id=summary.conversation_id, content=summary.content,
        created_at=now, updated_at=now, metadata=summary.metadata or {}
    )


@app.post("/summaries/batch", response_model=list[SummaryResponse], status_code=201, dependencies=[Depends(verify_api_key)])
def create_summaries_batch(batch: BatchSummaryCreate):
    """Create multiple summaries in batch."""
    if not batch.summaries:
        raise HTTPException(status_code=400, detail="No summaries provided")
    
    if len(batch.summaries) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 summaries per batch")

    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    
    # Prepare texts for batch embedding
    texts = [s.content for s in batch.summaries]
    vectors = embed_batch(texts)
    
    results = []
    try:
        for i, (summary, vector) in enumerate(zip(batch.summaries, vectors)):
            sid = generate_id()
            embedding = json.dumps(vector)
            
            cur.execute(
                "INSERT INTO summaries (id,conversation_id,content,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?)",
                (sid, summary.conversation_id, summary.content, embedding, now, now, json.dumps(summary.metadata or {}))
            )
            cur.execute("INSERT INTO summaries_fts (id,text) VALUES (?,?)", (sid, texts[i]))
            
            results.append(SummaryResponse(
                id=sid, conversation_id=summary.conversation_id, content=summary.content,
                created_at=now, updated_at=now, metadata=summary.metadata or {}
            ))
        
        conn.commit()
        metrics["summaries_created"] += len(results)
        logger.info(f"Created {len(results)} summaries in batch")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create summaries batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to create summaries batch")

    return results


@app.get("/summaries", response_model=list[SummaryResponse], dependencies=[Depends(verify_api_key)])
def list_summaries(
    conversation_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000)
):
    """List summaries with optional conversation filter."""
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
    return [
        SummaryResponse(
            id=r["id"], conversation_id=r["conversation_id"], content=r["content"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            metadata=json.loads(r["metadata"] or "{}")
        ) for r in rows
    ]


@app.delete("/summaries/{summary_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_summary(summary_id: str):
    """Delete a summary."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM summaries WHERE id=?", (summary_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Summary not found")
    try:
        cur.execute("DELETE FROM summaries_fts WHERE id=?", (summary_id,))
    except Exception:
        pass
    conn.commit()
    metrics["summaries_deleted"] += 1
    logger.info(f"Deleted summary: {summary_id}")


# ── Search ──────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search(req: SearchRequest):
    """
    Search for facts and summaries.
    Supports semantic, keyword, and hybrid search modes.
    """
    metrics["searches_performed"] += 1
    
    if req.mode == "keyword":
        return _keyword_search(req)
    elif req.mode == "hybrid":
        return _hybrid_search(req)
    else:  # semantic (default)
        return _semantic_search(req)


def _semantic_search(req: SearchRequest) -> SearchResponse:
    """Pure semantic search using vector similarity."""
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

    return SearchResponse(facts=facts, summaries=summaries)


def _keyword_search(req: SearchRequest) -> SearchResponse:
    """Pure keyword search using FTS5."""
    conn = get_db()
    cur = conn.cursor()

    # Search facts
    cur.execute("""
        SELECT f.* FROM facts f
        JOIN facts_fts fts ON f.id = fts.id
        WHERE facts_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (req.query, req.top_k))
    fact_rows = cur.fetchall()

    # Search summaries
    cur.execute("""
        SELECT s.* FROM summaries s
        JOIN summaries_fts fts ON s.id = fts.id
        WHERE summaries_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (req.query, req.top_k))
    summary_rows = cur.fetchall()

    facts = [
        FactResponse(
            id=r["id"], subject=r["subject"], predicate=r["predicate"], object=r["object"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            metadata=json.loads(r["metadata"] or "{}")
        ) for r in fact_rows
    ]

    summaries = [
        SummaryResponse(
            id=r["id"], conversation_id=r["conversation_id"], content=r["content"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            metadata=json.loads(r["metadata"] or "{}")
        ) for r in summary_rows
    ]

    return SearchResponse(facts=facts, summaries=summaries)


def _hybrid_search(req: SearchRequest) -> SearchResponse:
    """Hybrid search combining semantic and keyword results."""
    # Get semantic results
    semantic_results = _semantic_search(req)
    
    # Get keyword results
    keyword_results = _keyword_search(req)
    
    # Merge and deduplicate
    seen_fact_ids = set()
    seen_summary_ids = set()
    
    merged_facts = []
    merged_summaries = []
    
    # Add semantic results first (higher priority)
    for fact in semantic_results.facts:
        if fact.id not in seen_fact_ids:
            seen_fact_ids.add(fact.id)
            merged_facts.append(fact)
    
    for summary in semantic_results.summaries:
        if summary.id not in seen_summary_ids:
            seen_summary_ids.add(summary.id)
            merged_summaries.append(summary)
    
    # Add keyword results
    for fact in keyword_results.facts:
        if fact.id not in seen_fact_ids:
            seen_fact_ids.add(fact.id)
            merged_facts.append(fact)
    
    for summary in keyword_results.summaries:
        if summary.id not in seen_summary_ids:
            seen_summary_ids.add(summary.id)
            merged_summaries.append(summary)
    
    # Limit results
    merged_facts = merged_facts[:req.top_k]
    merged_summaries = merged_summaries[:req.top_k]
    
    return SearchResponse(facts=merged_facts, summaries=merged_summaries)


# ── Export/Import ───────────────────────────────────────────────────────────

@app.get("/export", response_model=ExportResponse, dependencies=[Depends(verify_api_key)])
def export_data():
    """Export all data."""
    conn = get_db()
    cur = conn.cursor()
    
    # Export facts
    cur.execute("SELECT * FROM facts ORDER BY created_at")
    fact_rows = cur.fetchall()
    facts = [
        {
            "id": r["id"],
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "metadata": json.loads(r["metadata"] or "{}")
        }
        for r in fact_rows
    ]
    
    # Export summaries
    cur.execute("SELECT * FROM summaries ORDER BY created_at")
    summary_rows = cur.fetchall()
    summaries = [
        {
            "id": r["id"],
            "conversation_id": r["conversation_id"],
            "content": r["content"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "metadata": json.loads(r["metadata"] or "{}")
        }
        for r in summary_rows
    ]
    
    stats = get_table_stats()
    
    logger.info(f"Exported {len(facts)} facts and {len(summaries)} summaries")
    
    return ExportResponse(
        facts=facts,
        summaries=summaries,
        exported_at=datetime.now().isoformat(),
        stats=stats
    )


@app.post("/import", response_model=ImportResponse, dependencies=[Depends(verify_api_key)])
def import_data(data: ImportRequest):
    """Import data with deduplication."""
    conn = get_db()
    cur = conn.cursor()
    
    imported_facts = 0
    imported_summaries = 0
    skipped_facts = 0
    skipped_summaries = 0
    
    try:
        # Import facts
        for fact in data.facts:
            # Check if exists
            cur.execute("SELECT id FROM facts WHERE id=?", (fact["id"],))
            if cur.fetchone():
                skipped_facts += 1
                continue
            
            # Re-embed if needed
            text = f"{fact['subject']} {fact['predicate']} {fact['object']}"
            vectors = embed([text])
            embedding = json.dumps(vectors[0])
            
            cur.execute(
                "INSERT INTO facts (id,subject,predicate,object,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?,?)",
                (fact["id"], fact["subject"], fact["predicate"], fact["object"], 
                 embedding, fact["created_at"], fact["updated_at"], json.dumps(fact.get("metadata", {})))
            )
            cur.execute("INSERT INTO facts_fts (id,text) VALUES (?,?)", (fact["id"], text))
            imported_facts += 1
        
        # Import summaries
        for summary in data.summaries:
            cur.execute("SELECT id FROM summaries WHERE id=?", (summary["id"],))
            if cur.fetchone():
                skipped_summaries += 1
                continue
            
            vectors = embed([summary["content"]])
            embedding = json.dumps(vectors[0])
            
            cur.execute(
                "INSERT INTO summaries (id,conversation_id,content,embedding,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?)",
                (summary["id"], summary["conversation_id"], summary["content"],
                 embedding, summary["created_at"], summary["updated_at"], json.dumps(summary.get("metadata", {})))
            )
            cur.execute("INSERT INTO summaries_fts (id,text) VALUES (?,?)", (summary["id"], summary["content"]))
            imported_summaries += 1
        
        conn.commit()
        logger.info(f"Imported {imported_facts} facts, {imported_summaries} summaries")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to import data: {e}")
        raise HTTPException(status_code=500, detail="Failed to import data")
    
    return ImportResponse(
        imported_facts=imported_facts,
        imported_summaries=imported_summaries,
        skipped_facts=skipped_facts,
        skipped_summaries=skipped_summaries
    )


# ── Metrics ─────────────────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Get service metrics."""
    stats = get_table_stats()
    uptime = time.time() - metrics["start_time"]
    
    return MetricsResponse(
        requests_total=metrics["requests_total"],
        requests_success=metrics["requests_success"],
        requests_error=metrics["requests_error"],
        facts_created=metrics["facts_created"],
        facts_deleted=metrics["facts_deleted"],
        summaries_created=metrics["summaries_created"],
        summaries_deleted=metrics["summaries_deleted"],
        searches_performed=metrics["searches_performed"],
        facts_count=stats["facts_count"],
        summaries_count=stats["summaries_count"],
        uptime_seconds=uptime
    )


# ── Maintenance ─────────────────────────────────────────────────────────────

@app.post("/maintenance/vacuum", dependencies=[Depends(verify_api_key)])
def vacuum():
    """Vacuum the database to reclaim space."""
    try:
        vacuum_database()
        logger.info("Database vacuumed successfully")
        return {"status": "ok", "message": "Database vacuumed"}
    except Exception as e:
        logger.error(f"Failed to vacuum database: {e}")
        raise HTTPException(status_code=500, detail="Failed to vacuum database")
