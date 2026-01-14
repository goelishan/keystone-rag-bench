from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import BM25Retriever

class RetrievalFactory:
    @staticmethod
    def create_retriever(strategy_config):
        if strategy_config.type == "dense":
            return DenseRetriever(strategy_config.subject)
        
        if strategy_config.type == "hybrid":
            return HybridRetriever(
                dense=DenseRetriever(strategy_config.dense_subject),
                sparse=BM25Retriever(strategy_config.sparse_subject),
                alpha=strategy_config.alpha
            )
        raise ValueError(f"Invalid retrieval strategy: {strategy_config.type}")