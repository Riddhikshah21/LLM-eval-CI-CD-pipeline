from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    provider: str
    name: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class RAGSettings(BaseModel):
    top_k: int = Field(default=3, gt=0)


class ModelConfig(BaseModel):
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


class EvaluationConfig(BaseModel):
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