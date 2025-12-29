import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from embeddings.openai_embedder import OpenAIEmbedder
from embeddings.normalize import l2_normalize
from retrieval.interfaces import Retriever, RetrievalConfig
from core.models import RetrievalResult


class FaissRetrieval(Retriever):

    def __init__(self, subject: str):
        self.subject = subject
        self._load_index()
        self._load_metadata()
        self.embedder = OpenAIEmbedder()

    def _load_index(self):
        index_path = Path("data/processed") / self.subject / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")
        self.index = faiss.read_index(str(index_path))

    def _load_metadata(self):
        metadata_path = Path("data/processed") / self.subject / "vector_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[RetrievalResult]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if config.top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Embed query
        query_vector = self.embedder.embed_documents([query])
        query_vector = l2_normalize(query_vector)

        scores, indices = self.index.search(query_vector, config.top_k)

        results: List[RetrievalResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            results.append(
                RetrievalResult(
                    score=float(score),
                    id=meta["id"],
                    source_id=meta["source_id"],
                    page=meta["page"],
                    chunk_index=meta["chunk_index"],
                )
            )

        return results
