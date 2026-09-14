from llm_eval.config import RAGSettings
from llm_eval.retrieval.bm25 import BM25Retriever
from llm_eval.retrieval.dense import (
    DenseRetriever,
    OpenAIEmbeddingProvider,
)
from llm_eval.retrieval.hybrid import HybridRetriever
from llm_eval.retrieval.utils import load_documents


def create_hybrid_retriever(
    config: RAGSettings,
    knowledge_base_path: str = "knowledge_base",
) -> HybridRetriever:
    documents = load_documents(knowledge_base_path)

    sparse = BM25Retriever(documents)

    embedding_provider = OpenAIEmbeddingProvider(model=config.embedding_model)

    dense = DenseRetriever(
        documents=documents,
        embedding_provider=embedding_provider,
    )

    return HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        sparse_top_k=config.sparse_top_k,
        dense_top_k=config.dense_top_k,
        rrf_k=config.rrf_k,
        sparse_weight=config.sparse_weight,
        dense_weight=config.dense_weight,
    )
