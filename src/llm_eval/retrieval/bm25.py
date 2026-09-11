from rank_bm25 import BM25Okapi

from llm_eval.retrieval.utils import tokenize
from llm_eval.schemas import Document, RetrievedDocument


class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("Documents cannot be empty")

        self.documents = documents

        corpus = [
            tokenize(document.content)
            for document in documents
        ]

        self.index = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = tokenize(query)

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            RetrievedDocument(
                id=self.documents[index].id,
                source=self.documents[index].source,
                content=self.documents[index].content,
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]