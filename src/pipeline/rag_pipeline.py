from typing import Optional

from core.telemetry import TelemetryRecorder
from core.models import RAGResponse
from retrieval.interfaces import Retriever, RetrievalConfig
from reranking.interfaces import Reranker


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[Reranker] = None
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.telemetry = TelemetryRecorder()

    def run(
        self,
        query: str,
        retrieval_config: RetrievalConfig
    ) -> RAGResponse:
        raise NotImplementedError("Pipeline execution not implemented yet")
