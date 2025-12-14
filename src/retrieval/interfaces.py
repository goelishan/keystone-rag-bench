from abc import ABC, abstractmethod
from typing import List

from core.models import RetrievalResult


class RetrievalConfig:
    def __init__(self, top_k: int, alpha: float | None = None):
        self.top_k = top_k
        self.alpha = alpha


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        pass
