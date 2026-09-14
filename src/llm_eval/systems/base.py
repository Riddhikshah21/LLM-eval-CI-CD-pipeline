from typing import Any, Protocol

from pydantic import BaseModel, Field


class SystemResult(BaseModel):
    response: str

    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class SystemUnderTest(Protocol):
    def run(
        self,
        input_data: dict[str, Any],
    ) -> SystemResult:
        ...