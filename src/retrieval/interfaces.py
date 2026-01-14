from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from core.models import RetrievalResult

@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int=5


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        pass
