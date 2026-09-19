import json
from pathlib import Path

from llm_eval.schemas import GoldenExample


def load_golden_dataset(
    path: str | Path = "data/golden_dataset.jsonl",
) -> list[GoldenExample]:
    dataset_path = Path(path)

    examples = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            examples.append(GoldenExample.model_validate(json.loads(line)))

    return examples


def test_load_golden_dataset() -> None:
    dataset = load_golden_dataset()

    assert len(dataset) == 3
    assert dataset[0].id == "refund_001"
    assert dataset[0].input["question"] == ("How long do I have to return an item?")
    assert dataset[2].metadata["should_abstain"] is True
