from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    source: str
    content: str


class GoldenExample(BaseModel):
    id: str

    input: dict[str, Any]

    expected_output: dict[str, Any] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    id: str
    source: str
    content: str
    score: float


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class PipelineResult(BaseModel):
    question: str
    answer: str
    retrieved_context: list[str]
    retrieved_sources: list[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
