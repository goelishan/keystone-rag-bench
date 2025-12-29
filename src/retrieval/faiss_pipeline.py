from retrieval.faiss_retrieval import FaissRetrieval
from retrieval.interfaces import RetrievalConfig

retriever = FaissRetrieval(
    subject="cloud_devops_docs_v1"
)

config = RetrievalConfig(top_k=5)

results = retriever.retrieve(
    "What is the AWS Well-Architected Framework?",
    config=config
)

for r in results:
    print(
        f"score={r.score:.3f}",
        f"source={r.source_id}",
        f"page={r.page}",
        f"chunk={r.chunk_index}"
    )
