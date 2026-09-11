from langfuse import get_client

from llm_eval.dataset import load_golden_dataset


DATASET_NAME = "llm-evaluation-golden"


def sync_dataset() -> None:
    langfuse = get_client()

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Golden dataset for LLM/RAG regression testing",
    )

    examples = load_golden_dataset()

    for example in examples:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=example.id,
            input={
                "question": example.question,
            },
            expected_output={
                "answer": example.expected_answer,
            },
            metadata={
                "expected_sources": example.expected_sources,
                "should_abstain": example.should_abstain,
                "category": example.category,
                "tags": example.tags,
            },
        )

    langfuse.flush()


if __name__ == "__main__":
    sync_dataset()