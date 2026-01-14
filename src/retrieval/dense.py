import json
from pathlib import Path
from typing import List

import faiss

from embeddings.normalize import l2_normalize
from retrieval.interfaces import Retriever, RetrievalConfig
from core.models import RetrievalResult


class DenseRetriever(Retriever):
    """
    Dense retrieval using FAISS + cosine similarity.
    Embedder is injected to keep retrieval testable and offline-safe.
    """

    def __init__(self, subject: str, embedder=None):
        self.subject = subject
        self.embedder = embedder  # injected dependency
        self._load_index()
        self._load_chunks()

    def _load_index(self):
        index_path = Path("data/processed") / self.subject / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")
        self.index = faiss.read_index(str(index_path))

    def _load_chunks(self):
        chunks_path = Path("data/processed") / self.subject / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, config: RetrievalConfig) -> List[RetrievalResult]:
        if self.embedder is None:
            raise RuntimeError(
                "DenseRetriever requires an embedder to perform retrieval"
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if config.top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Embed query (query-time only)
        query_vector = self.embedder.embed_query(query)
        query_vector = l2_normalize(query_vector)

        scores, indices = self.index.search(query_vector, config.top_k)

        results: List[RetrievalResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            meta = chunk["metadata"]

            results.append(
                RetrievalResult(
                    score=float(score),
                    id=chunk["id"],
                    source_id=meta["source_id"],
                    page=meta["page"],
                    chunk_index=meta["chunk_index"],
                )
            )

        return results
