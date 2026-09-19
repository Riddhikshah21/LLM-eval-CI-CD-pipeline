from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PricingSettings(BaseModel):
    input_per_1m_tokens_usd: float = Field(ge=0)
    output_per_1m_tokens_usd: float = Field(ge=0)


class ModelSettings(BaseModel):
    provider: str
    name: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    pricing: PricingSettings


class RAGSettings(BaseModel):
    top_k: int = Field(default=3, gt=0)

    sparse_top_k: int = Field(default=10, gt=0)
    dense_top_k: int = Field(default=10, gt=0)

    embedding_model: str = "text-embedding-3-small"

    rrf_k: int = Field(default=60, gt=0)

    sparse_weight: float = Field(default=0.5, ge=0.0)
    dense_weight: float = Field(default=0.5, ge=0.0)


class ApplicationSettings(BaseModel):
    type: str


class ModelConfig(BaseModel):
    application: ApplicationSettings
    model: ModelSettings
    rag: RAGSettings


class LatencyThresholds(BaseModel):
    p95_max_ms: int = Field(gt=0)
    max_regression_percent: float = Field(ge=0)


class CostThresholds(BaseModel):
    max_mean_per_query_usd: float = Field(ge=0)


class EvaluationThresholds(BaseModel):
    answer_relevancy: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    max_hallucination_rate: float = Field(ge=0, le=1)
    latency: LatencyThresholds
    cost: CostThresholds
    correctness: float = Field(ge=0, le=1)


class JudgeSettings(BaseModel):
    model: str


class EvaluationConfig(BaseModel):
    judge: JudgeSettings
    thresholds: EvaluationThresholds


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model_config(path: str | Path = "config/model.yaml") -> ModelConfig:
    return ModelConfig.model_validate(_load_yaml(Path(path)))


def load_evaluation_config(
    path: str | Path = "config/evaluation.yaml",
) -> EvaluationConfig:
    return EvaluationConfig.model_validate(_load_yaml(Path(path)))
