from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.interfaces import RetrievalConfig

import numpy as np

class FakeEmbedder:
    def embed_query(self, text: str):
        # Must match FAISS index dimension
        return np.ones((1, 1536), dtype="float32")


def test_dense_retrieval():
    retriever = DenseRetriever(
    subject="cloud_devops_docs_v1",
    embedder=FakeEmbedder(),
)


    results = retriever.retrieve(
        query="AWS Well-Architected Framework",
        config=RetrievalConfig(top_k=5),
    )

    assert len(results) > 0
    for r in results:
        print(r)
        assert r.id
        assert isinstance(r.score, float)


def test_bm25_retrieval():
    retriever = BM25Retriever(subject="cloud_devops_docs_v1")

    results = retriever.retrieve(
        query="AWS Well-Architected Framework",
        config=RetrievalConfig(top_k=5),
    )

    assert len(results) > 0
    for r in results:
        print(r)
        assert 0.0 <= r.score <= 1.0


def test_hybrid_retrieval():
    fake = FakeEmbedder()

    dense = DenseRetriever("cloud_devops_docs_v1",embedder=fake)
    sparse = BM25Retriever("cloud_devops_docs_v1")

    retriever = HybridRetriever(
        dense=dense,
        sparse=sparse,
        alpha=0.6,
    )

    results = retriever.retrieve(
        query="AWS Well-Architected Framework reliability",
        config=RetrievalConfig(top_k=5),
    )

    assert len(results) > 0
