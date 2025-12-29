from dataclasses import dataclass
from typing import Any, List, Dict


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    score: float
    id: str
    source_id: str
    page: int
    chunk_index: int


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    latency: Dict[str, float]
