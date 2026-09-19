def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_per_1m_tokens_usd: float,
    output_per_1m_tokens_usd: float,
) -> float:
    input_cost = (input_tokens / 1_000_000) * input_per_1m_tokens_usd

    output_cost = (output_tokens / 1_000_000) * output_per_1m_tokens_usd

    return input_cost + output_cost
