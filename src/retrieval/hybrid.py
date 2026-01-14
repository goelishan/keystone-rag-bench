from typing import Dict, List

from retrieval.interfaces import Retriever, RetrievalConfig
from core.models import RetrievalResult


class HybridRetriever(Retriever):
    """
    Score-level fusion of dense and sparse retrieval.
    """

    def __init__(self, dense: Retriever, sparse: Retriever, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query: str, config: RetrievalConfig) -> List[RetrievalResult]:
        dense_results = self.dense.retrieve(query, config)
        sparse_results = self.sparse.retrieve(query, config)

        fused: Dict[str, RetrievalResult] = {}

        # Add dense results
        for r in dense_results:
            fused[r.id] = RetrievalResult(
                id=r.id,
                source_id=r.source_id,
                page=r.page,
                chunk_index=r.chunk_index,
                score=self.alpha * r.score,
            )

        # Merge sparse results
        for r in sparse_results:
            if r.id in fused:
                fused[r.id].score += (1 - self.alpha) * r.score
            else:
                fused[r.id] = RetrievalResult(
                    id=r.id,
                    source_id=r.source_id,
                    page=r.page,
                    chunk_index=r.chunk_index,
                    score=(1 - self.alpha) * r.score,
                )

        results = sorted(
            fused.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return results[: config.top_k]
