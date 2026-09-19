from llm_eval.baseline import regression_percent


def test_latency_regression() -> None:
    result = regression_percent(
        current=1200,
        baseline=1000,
    )

    assert result == 20.0


def test_latency_improvement() -> None:
    result = regression_percent(
        current=800,
        baseline=1000,
    )

    assert result == -20.0
