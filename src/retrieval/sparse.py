import json
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingImports]

from retrieval.interfaces import Retriever, RetrievalConfig
from core.models import RetrievalResult

class BM25Retriever(Retriever):
    """
    Sparse retrieval using BM25 over chunk text.
    Scores normalized to [0, 1].
    """

    def __init__(self, subject: str):
        self.subject = subject
        self._load_chunks()
        self._build_bm25()

    def _load_chunks(self):
        path = Path("data/processed") / self.subject / "chunks.json"
        if not path.exists():
            raise FileNotFoundError(f"chunks.json not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.documents = [c["text"] for c in self.chunks]

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _build_bm25(self):
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)


    def retrieve(self, query: str, config: RetrievalConfig) -> List[RetrievalResult]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_k = min(config.top_k, len(scores))
        top_indices = np.argsort(scores)[:-top_k][::-1]

        top_scores = scores[top_indices]
        max_s,min_s=float(np.max(top_scores)),float(np.min(top_scores))

        if max_s == min_s:
            norm_scores=np.ones_like(top_scores)
        else:
            norm_scores=(top_scores-min_s)/(max_s-min_s)
        
        results:List[RetrievalResult]=[]

        for score,idx in zip(norm_scores,top_indices):
            chunk=self.chunks[idx]
            meta=chunk["metadata"]

            results.append(
                RetrievalResult(
                    score=float(score),
                    id=chunk["id"],
                    source_id=meta["source_id"],
                    page=meta["page"],
                    chunk_index=meta["chunk_index"]
                )
            )
        return results