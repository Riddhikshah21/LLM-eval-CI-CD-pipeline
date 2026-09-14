from typing import ClassVar

import numpy as np

from llm_eval.retrieval.dense import DenseRetriever
from llm_eval.schemas import Document


class FakeEmbeddingProvider:
    vectors: ClassVar[dict[str, list[float]]] = {
        "refund policy return item": [1.0, 0.0, 0.0],
        "shipping delivery package": [0.0, 1.0, 0.0],
        "password account reset": [0.0, 0.0, 1.0],
        "return refund": [1.0, 0.0, 0.0],
    }

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return np.asarray(
            [self.vectors[text] for text in texts],
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        return np.asarray(
            [self.vectors[text]],
            dtype=np.float32,
        )


def test_dense_retrieval() -> None:
    documents = [
        Document(
            id="1",
            source="refunds.md",
            content="refund policy return item",
        ),
        Document(
            id="2",
            source="shipping.md",
            content="shipping delivery package",
        ),
        Document(
            id="3",
            source="accounts.md",
            content="password account reset",
        ),
    ]

    retriever = DenseRetriever(
        documents=documents,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = retriever.retrieve(
        "return refund",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].source == "refunds.md"
