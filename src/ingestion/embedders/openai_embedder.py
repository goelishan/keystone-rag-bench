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
        # model: OpenAI embedding model to use (e.g., "text-embedding-3-small")
        # batch_size: how many texts to send per API request
        # client: optional, custom OpenAI client instance
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.client = client or OpenAI()
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # texts: list of text strings to embed

        # Defensive: fail if no input texts
        if not texts:
            raise ValueError("embed_texts received an empty input list")

        # Defensive: type and content checking of each input
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"Text at index {idx} is not a string")
            if not t.strip():
                raise ValueError(f"Empty or whitespace-only text at index {idx}")

        all_embeddings: List[List[float]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
            except Exception as e:
                # Meaning: Exception means OpenAI API call failed (network, rate-limit, etc.)
                raise RuntimeError(
                    f"OpenAI embedding request failed "
                    f"(model={self.model}, "
                    f"batch_start={batch_start}, "
                    f"batch_size={len(batch)}): {e}"
                ) from e

            # Meaning: Ensure the API gave a list of data as expected
            if not hasattr(response, 'data') or not isinstance(response.data, list):
                raise RuntimeError(
                    f"OpenAI API response missing 'data' attribute or not a list (batch_start={batch_start})"
                )
            try:
                # Each element in response.data must have an 'embedding' attribute
                batch_embeddings = [item.embedding for item in response.data]
            except Exception as e:
                # Meaning: API returned malformed data structure
                raise RuntimeError(
                    f"Failed to extract embeddings from response (batch_start={batch_start}): {e}"
                ) from e

            # Meaning: API returned fewer/more embeddings than requested
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch in batch starting at {batch_start}: "
                    f"received {len(batch_embeddings)}, expected {len(batch)}"
                )

            all_embeddings.extend(batch_embeddings)

        # Meaning: Defensive - no content returned from any batch
        if not all_embeddings:
            raise RuntimeError("No embeddings returned from OpenAI API")

        # Meaning: Defensive - count should exactly match requested input texts
        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                f"Total embedding count mismatch: "
                f"{len(all_embeddings)} embeddings for {len(texts)} texts"
            )

        # Meaning: all returned embeddings must be the same length (dimensionality)
        embedding_dim = len(all_embeddings[0])
        for i, emb in enumerate(all_embeddings):
            if not isinstance(emb, (list, np.ndarray)):
                raise ValueError(
                    f"Embedding at index {i} is not a list or array: type={type(emb)}"
                )
            if len(emb) != embedding_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension at index {i}: "
                    f"expected {embedding_dim}, got {len(emb)}"
                )

        try:
            # Meaning: convert to numpy for downstream use and normalization
            embeddings = np.array(all_embeddings, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to convert embeddings to np.ndarray: {e}") from e

        # Meaning: Defensive - check for bad values in embedding vectors
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("NaN or infinite values found in embeddings array")

        # Meaning: cannot normalize a zero-vector; important for later similarity search
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero-norm embedding(s) found; cannot normalize")

        try:
            # Meaning: L2 normalization so embeddings are on a unit hypersphere
            embeddings = self.l2_normalize(embeddings)
        except Exception as e:
            raise RuntimeError(f"L2 normalization failed: {e}") from e

        # Meaning: Return a matrix (num_texts, embedding_dim) of float32 normalized embeddings
        return embeddings
