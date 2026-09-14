import json
from pathlib import Path

from llm_eval.schemas import GoldenExample


def load_golden_dataset(
    path: str | Path = "data/golden_dataset.jsonl",
) -> list[GoldenExample]:
    examples: list[GoldenExample] = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            examples.append(
                GoldenExample.model_validate_json(line)
            )

    return examples