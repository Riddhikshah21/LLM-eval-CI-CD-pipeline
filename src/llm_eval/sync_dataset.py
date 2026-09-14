from langfuse import get_client

from llm_eval.dataset import load_golden_dataset


DATASET_NAME = "llm-evaluation-golden"


def sync_dataset() -> None:
    langfuse = get_client()

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Golden dataset for AI application regression testing",
    )

    for example in load_golden_dataset():
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=example.id,
            input=example.input,
            expected_output=example.expected_output,
            metadata=example.metadata,
        )

    langfuse.flush()


if __name__ == "__main__":
    sync_dataset()