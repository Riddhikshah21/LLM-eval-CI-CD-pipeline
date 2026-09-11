from llm_eval.retriever import BM25Retriever


def test_retriever_returns_refund_document() -> None:
    retriever = BM25Retriever("knowledge_base")

    results = retriever.retrieve(
        "How long do I have to return an item?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].source == "refunds.md"