"""Pydantic schemas for REFRAG RAG system."""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model."""
    id: str
    source: str  # e.g., "local_files", "aicar_logs", "rules"
    text: str
    metadata: Optional[Dict[str, str]] = None


class Chunk(BaseModel):
    """Chunk model with token information."""
    id: str
    document_id: str
    text: str
    token_count: int
    start_offset: int
    end_offset: int
    metadata: Optional[Dict[str, str]] = None


class RetrievedChunk(BaseModel):
    """Retrieved chunk with relevance score."""
    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: int = Field(default=4, ge=1, le=32)  # Further reduced from 5 to 4 for faster responses


class QueryResponse(BaseModel):
    """Query response model with REFRAG metadata."""
    answer: str
    used_chunks: List[Chunk]
    compression_decisions: Dict[str, str]  # chunk_id -> "COMPRESS" | "EXPAND"
    prompt_token_count: int
    llm_latency_ms: float

