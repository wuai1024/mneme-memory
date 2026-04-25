from pydantic import BaseModel
from typing import Optional


class FactCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    metadata: Optional[dict] = None


class FactResponse(BaseModel):
    id: str
    subject: str
    predicate: str
    object: str
    created_at: str
    updated_at: str
    metadata: dict = {}


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


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    facts: list[FactResponse] = []
    summaries: list[SummaryResponse] = []


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_dimension: int
