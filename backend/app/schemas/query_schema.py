from pydantic import BaseModel, Field
from typing import List, Optional

from app.schemas.source_schema import SourceItem


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of retrieved chunks")
    file_name: Optional[str] = Field(default=None, description="Optional file name to restrict retrieval")


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    file_name: Optional[str] = None
    page: Optional[int] = None
    distance: Optional[float] = None
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: List[SourceItem]
    results: List[SearchResult]
    suggested_questions: List[str] = Field(default_factory=list)
