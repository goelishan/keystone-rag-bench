from dataclasses import dataclass
from typing import Any,List,Dict,Optional


@dataclass
class chunk:
  id: str
  text: str
  metadata=Dict[str, Any]


@dataclass
class RetrivalResult:
  chunk: chunk
  score: float
  retriever_name: str
  rank: Optional[int] = None


@dataclass
class RAGResponse:
  answer: str
  sources: List[Dict[str, Any]]
  metrics: Dict[str, Any]
  latency: Dict[str, float]
