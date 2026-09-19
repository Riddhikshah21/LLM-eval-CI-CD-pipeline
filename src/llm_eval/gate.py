import json
from pathlib import Path

from llm_eval.config import load_evaluation_config

REPORT_PATH = Path("reports/evaluation.json")
from llm_eval.baseline import load_baseline, regression_percent

BASELINE_PATH = Path("reports/baseline/evaluation.json")


def run_gate() -> None:
    config = load_evaluation_config()

    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Evaluation report not found: {REPORT_PATH}")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    metrics = report["metrics"]

    quality = metrics["quality"]
    latency = metrics["latency"]
    cost = metrics["cost"]

    thresholds = config.thresholds

    failures: list[str] = []

    _check_minimum(
        failures,
        name="correctness",
        actual=quality["correctness"],
        minimum=thresholds.correctness,
    )

    _check_minimum(
        failures,
        name="answer relevancy",
        actual=quality["answer_relevancy"],
        minimum=thresholds.answer_relevancy,
    )

    _check_minimum(
        failures,
        name="faithfulness",
        actual=quality["faithfulness"],
        minimum=thresholds.faithfulness,
    )

    _check_maximum(
        failures,
        name="hallucination rate",
        actual=quality["hallucination_rate"],
        maximum=thresholds.max_hallucination_rate,
    )

    _check_maximum(
        failures,
        name="p95 latency",
        actual=latency["p95_ms"],
        maximum=thresholds.latency.p95_max_ms,
        unit="ms",
    )
    baseline = load_baseline(BASELINE_PATH)

    if baseline is not None:
        baseline_p95 = baseline["metrics"]["latency"]["p95_ms"]
        current_p95 = latency["p95_ms"]

        if baseline_p95 is not None and current_p95 is not None:
            latency_regression = regression_percent(
                current=current_p95,
                baseline=baseline_p95,
            )

            print(f"P95 baseline:      {baseline_p95:.2f} ms")
            print(f"P95 regression:    {latency_regression:.2f}%")

            if latency_regression > thresholds.latency.max_regression_percent:
                failures.append(
                    "p95 latency regression: "
                    f"{latency_regression:.2f}% > "
                    f"{thresholds.latency.max_regression_percent:.2f}%"
                )
    else:
        print("Baseline: not available (regression check skipped)")

    # Cost is optional until cost calculation is implemented.
    if cost["mean_per_query_usd"] is not None:
        _check_maximum(
            failures,
            name="mean cost per query",
            actual=cost["mean_per_query_usd"],
            maximum=(thresholds.cost.max_mean_per_query_usd),
            unit="USD",
        )

    _print_summary(
        quality=quality,
        latency=latency,
        cost=cost,
    )

    if failures:
        print("\nEvaluation gate: FAIL")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    print("\nEvaluation gate: PASS")


def _check_minimum(
    failures: list[str],
    name: str,
    actual: float | None,
    minimum: float,
) -> None:
    if actual is None:
        failures.append(f"{name}: metric is missing")
        return

    if actual < minimum:
        failures.append(f"{name}: {actual:.4f} < {minimum:.4f}")


def _check_maximum(
    failures: list[str],
    name: str,
    actual: float | None,
    maximum: float,
    unit: str = "",
) -> None:
    if actual is None:
        failures.append(f"{name}: metric is missing")
        return

    suffix = f" {unit}" if unit else ""

    if actual > maximum:
        failures.append(f"{name}: {actual:.4f}{suffix} > {maximum:.4f}{suffix}")


def _print_summary(
    quality: dict,
    latency: dict,
    cost: dict,
) -> None:
    print("\nEvaluation summary")
    print("------------------")

    print(f"Correctness:       {quality['correctness']:.4f}")
    print(f"Answer relevancy:  {quality['answer_relevancy']:.4f}")
    print(f"Faithfulness:      {quality['faithfulness']:.4f}")
    print(f"Hallucination:     {quality['hallucination_rate']:.2%}")
    print(f"P50 latency:       {latency['p50_ms']:.2f} ms")
    print(f"P95 latency:       {latency['p95_ms']:.2f} ms")

    if cost["mean_per_query_usd"] is not None:
        print(f"Cost/query:        ${cost['mean_per_query_usd']:.6f}")


if __name__ == "__main__":
    run_gate()
