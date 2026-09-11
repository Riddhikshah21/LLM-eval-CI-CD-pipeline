from llm_eval.retrieval.bm25 import BM25Retriever
from llm_eval.schemas import Document


def test_bm25_retrieval() -> None:
    documents = [
        Document(
            id="1",
            source="refunds.md",
            content="Customers can return items within 30 days.",
        ),
        Document(
            id="2",
            source="shipping.md",
            content="Standard shipping takes 3 to 5 business days.",
        ),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "return item",
        top_k=1,
    )

    assert results[0].source == "refunds.md"