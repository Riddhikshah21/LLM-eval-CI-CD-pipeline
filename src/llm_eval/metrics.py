from statistics import mean

import numpy as np


def aggregate_metrics(
    item_results,
    faithfulness_threshold: float,
) -> dict:
    scores = {
        "correctness": [],
        "answer_relevancy": [],
        "faithfulness": [],
        "abstention_accuracy": [],
    }

    latencies = []
    costs = []
    hallucinated = 0

    for item_result in item_results:
        output = item_result.output or {}

        if output.get("latency_ms") is not None:
            latencies.append(float(output["latency_ms"]))

        if output.get("cost_usd") is not None:
            costs.append(float(output["cost_usd"]))

        faithfulness_score = None

        for evaluation in item_result.evaluations:
            if evaluation.value is None:
                continue

            if evaluation.name in scores:
                scores[evaluation.name].append(
                    float(evaluation.value)
                )

            if evaluation.name == "faithfulness":
                faithfulness_score = float(
                    evaluation.value
                )

        if (
            faithfulness_score is not None
            and faithfulness_score
            < faithfulness_threshold
        ):
            hallucinated += 1

    faithfulness_count = len(
        scores["faithfulness"]
    )

    hallucination_rate = (
        hallucinated / faithfulness_count
        if faithfulness_count
        else None
    )

    return {
        "quality": {
            "correctness": _average(
                scores["correctness"]
            ),
            "answer_relevancy": _average(
                scores["answer_relevancy"]
            ),
            "faithfulness": _average(
                scores["faithfulness"]
            ),
            "abstention_accuracy": _average(
                scores["abstention_accuracy"]
            ),
            "hallucination_rate": (
                hallucination_rate
            ),
        },
        "latency": {
            "p50_ms": _percentile(
                latencies,
                50,
            ),
            "p95_ms": _percentile(
                latencies,
                95,
            ),
        },
        "cost": {
            "mean_per_query_usd": (
                _average(costs)
            ),
            "total_usd": (
                float(sum(costs))
                if costs
                else None
            ),
        },
        "total_examples": len(item_results),
    }


def _average(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(mean(values))


def _percentile(
    values: list[float],
    percentile: int,
) -> float | None:
    if not values:
        return None

    return float(
        np.percentile(
            values,
            percentile,
        )
    )