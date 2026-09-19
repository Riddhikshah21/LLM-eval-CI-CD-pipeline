import json
from pathlib import Path


def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def regression_percent(
    current: float,
    baseline: float,
) -> float:
    if baseline == 0:
        return 0.0

    return ((current - baseline) / baseline) * 100
