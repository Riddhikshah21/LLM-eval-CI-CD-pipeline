from llm_eval.config import load_evaluation_config, load_model_config


def test_load_model_config() -> None:
    config = load_model_config()

    assert config.model.provider == "openai"
    assert config.model.name == "gpt-4.1-mini"
    assert config.rag.top_k == 3


def test_load_evaluation_config() -> None:
    config = load_evaluation_config()

    assert config.thresholds.answer_relevancy == 0.80
    assert config.thresholds.correctness == 0.80
    assert config.thresholds.faithfulness == 0.80
    assert config.thresholds.max_hallucination_rate == 0.34
    assert config.thresholds.latency.p95_max_ms == 5000
