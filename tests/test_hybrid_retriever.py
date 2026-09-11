from llm_eval.retrieval.hybrid import HybridRetriever
from llm_eval.schemas import RetrievedDocument


class FakeRetriever:
    def __init__(
        self,
        results: list[RetrievedDocument],
    ) -> None:
        self.results = results

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedDocument]:
        return self.results[:top_k]


def test_hybrid_retrieval() -> None:
    refunds = RetrievedDocument(
        id="refund",
        source="refunds.md",
        content="Refund policy",
        score=1.0,
    )

    shipping = RetrievedDocument(
        id="shipping",
        source="shipping.md",
        content="Shipping policy",
        score=0.8,
    )

    accounts = RetrievedDocument(
        id="accounts",
        source="accounts.md",
        content="Account policy",
        score=0.9,
    )

    sparse = FakeRetriever(
        [refunds, shipping]
    )

    dense = FakeRetriever(
        [refunds, accounts]
    )

    retriever = HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
    )

    results = retriever.retrieve(
        "refund question",
        top_k=2,
    )

    assert results[0].source == "refunds.md"