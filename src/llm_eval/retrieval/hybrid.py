from collections import defaultdict

from langfuse import observe

from llm_eval.retrieval.bm25 import BM25Retriever
from llm_eval.retrieval.dense import DenseRetriever
from llm_eval.schemas import RetrievedDocument


class HybridRetriever:
    def __init__(
        self,
        sparse_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        sparse_top_k: int = 10,
        dense_top_k: int = 10,
        rrf_k: int = 60,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever

        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k

        self.rrf_k = rrf_k

        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    @observe(name="hybrid-retrieval", as_type="retriever")
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedDocument]:
        sparse_results = self.sparse_retriever.retrieve(
            query=query,
            top_k=self.sparse_top_k,
        )

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=self.dense_top_k,
        )

        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, RetrievedDocument] = {}

        self._add_rrf_scores(
            results=sparse_results,
            scores=scores,
            documents=documents,
            weight=self.sparse_weight,
        )

        self._add_rrf_scores(
            results=dense_results,
            scores=scores,
            documents=documents,
            weight=self.dense_weight,
        )

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        return [
            RetrievedDocument(
                id=document_id,
                source=documents[document_id].source,
                content=documents[document_id].content,
                score=scores[document_id],
            )
            for document_id in ranked_ids[:top_k]
        ]

    def _add_rrf_scores(
        self,
        results: list[RetrievedDocument],
        scores: dict[str, float],
        documents: dict[str, RetrievedDocument],
        weight: float,
    ) -> None:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            documents[result.id] = result

            scores[result.id] += (
                weight / (self.rrf_k + rank)
            )