from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None

class Source(BaseModel):
    source: str
    section: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str

class ResetRequest(BaseModel):
    session_id: str

class ResetResponse(BaseModel):
    status: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    collection_stats: dict