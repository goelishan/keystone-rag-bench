from abc import ABC, abstractmethod
from typing import List

from core.models import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        pass
