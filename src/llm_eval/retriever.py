import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from llm_eval.schemas import RetrievedDocument


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, knowledge_base_path: str | Path) -> None:
        self.knowledge_base_path = Path(knowledge_base_path)

        self.documents = [
            (file.name, file.read_text(encoding="utf-8"))
            for file in sorted(self.knowledge_base_path.glob("*.md"))
        ]

        tokenized_documents = [
            _tokenize(content)
            for _, content in self.documents
        ]

        self.index = BM25Okapi(tokenized_documents)

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[RetrievedDocument]:
        scores = self.index.get_scores(_tokenize(question))

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RetrievedDocument(
                source=document[0],
                content=document[1],
                score=float(score),
            )
            for document, score in ranked[:top_k]
        ]