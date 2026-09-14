from llm_eval.cost import calculate_cost


def test_calculate_cost() -> None:
    cost = calculate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_per_1m_tokens_usd=0.40,
        output_per_1m_tokens_usd=1.60,
    )

    assert cost == 2.0
