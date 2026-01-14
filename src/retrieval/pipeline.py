from retrieval.registry import RetrieverFactory
from retrieval.interfaces import RetrievalConfig

class RetrievalPipeline:
    """
    End-to-end retrieval orchestration.
    """

    def __init__(self, strategy_config):
        self.retriever = RetrieverFactory.create(strategy_config)

    def run(self, query: str, top_k: int):
        config = RetrievalConfig(top_k=top_k)
        return self.retriever.retrieve(query, config)

    