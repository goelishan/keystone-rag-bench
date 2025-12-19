from typing import List, Optional
import numpy as np
from openai import OpenAI

from ingestion.embedders.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 32,
        client: Optional[OpenAI] = None
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.client = client or OpenAI()
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # ---- Input validation ----
        if not texts:
            raise ValueError("embed_texts received an empty input list")

        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"Text at index {idx} is not a string")
            if not t.strip():
                raise ValueError(f"Empty or whitespace-only text at index {idx}")

        all_embeddings: List[List[float]] = []

        # ---- Batch processing ----
        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
            except Exception as e:
                raise RuntimeError(
                    f"OpenAI embedding request failed "
                    f"(model={self.model}, "
                    f"batch_start={batch_start}, "
                    f"batch_size={len(batch)})"
                ) from e

            batch_embeddings = [item.embedding for item in response.data]

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch in batch starting at {batch_start}: "
                    f"received {len(batch_embeddings)}, expected {len(batch)}"
                )

            all_embeddings.extend(batch_embeddings)

        # ---- Final integrity checks ----
        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                f"Total embedding count mismatch: "
                f"{len(all_embeddings)} embeddings for {len(texts)} texts"
            )

        # ---- Dimensionality validation ----
        embedding_dim = len(all_embeddings[0])
        for i, emb in enumerate(all_embeddings):
            if len(emb) != embedding_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension at index {i}: "
                    f"expected {embedding_dim}, got {len(emb)}"
                )

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # ---- Mandatory normalization ----
        embeddings = self.l2_normalize(embeddings)

        return embeddings
