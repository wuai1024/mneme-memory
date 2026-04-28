from pydantic import BaseModel
from typing import Optional


class FactCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    metadata: Optional[dict] = None


class FactUpdate(BaseModel):
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    metadata: Optional[dict] = None


class FactResponse(BaseModel):
    id: str
    subject: str
    predicate: str
    object: str
    created_at: str
    updated_at: str
    metadata: dict = {}


class BatchFactCreate(BaseModel):
    facts: list[FactCreate]


class BatchFactDelete(BaseModel):
    ids: list[str]


class SummaryCreate(BaseModel):
    conversation_id: str
    content: str
    metadata: Optional[dict] = None


class SummaryResponse(BaseModel):
    id: str
    conversation_id: str
    content: str
    created_at: str
    updated_at: str
    metadata: dict = {}


class BatchSummaryCreate(BaseModel):
    summaries: list[SummaryCreate]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "semantic"  # semantic, keyword, hybrid


class SearchResponse(BaseModel):
    facts: list[FactResponse] = []
    summaries: list[SummaryResponse] = []


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_dimension: int
    version: str = "1.1.0"


class ExportResponse(BaseModel):
    facts: list[dict]
    summaries: list[dict]
    exported_at: str
    stats: dict


class ImportRequest(BaseModel):
    facts: list[dict] = []
    summaries: list[dict] = []


class ImportResponse(BaseModel):
    imported_facts: int
    imported_summaries: int
    skipped_facts: int
    skipped_summaries: int


class MetricsResponse(BaseModel):
    requests_total: int
    requests_success: int
    requests_error: int
    facts_created: int
    facts_deleted: int
    summaries_created: int
    summaries_deleted: int
    searches_performed: int
    facts_count: int
    summaries_count: int
    uptime_seconds: float
