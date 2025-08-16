from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    doc_ids: List[str]

class ChatRequest(BaseModel):
    question: str
    k: int = 5

class ChatAnswer(BaseModel):
    answer: str
    sources: List[dict]

class CompareRequest(BaseModel):
    doc_ids: List[str]
    k: int = 5

class TopicsResponse(BaseModel):
    n_clusters: int
    labels: List[int]
    samples: List[dict]

class SummariesResponse(BaseModel):
    summaries: List[dict]
