import json
from pathlib import Path

from llm_eval.schemas import GoldenExample


def load_golden_dataset(
    path: str | Path = "data/golden_dataset.jsonl",
) -> list[GoldenExample]:
    dataset_path = Path(path)

    examples: list[GoldenExample] = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            data = json.loads(line)

            examples.append(
                GoldenExample.model_validate(data)
            )

    return examples