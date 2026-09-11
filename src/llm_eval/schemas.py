from pydantic import BaseModel, Field



class Document(BaseModel):
    id: str
    source: str
    content: str

class GoldenExample(BaseModel):
    id: str
    question: str
    expected_answer: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    category: str
    tags: list[str] = Field(default_factory=list)

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