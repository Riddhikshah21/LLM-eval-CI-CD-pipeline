from pydantic import BaseModel, Field


class GoldenExample(BaseModel):
    id: str
    question: str
    expected_answer: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    category: str
    tags: list[str] = Field(default_factory=list)

class RetrievedDocument(BaseModel):
    source: str
    content: str
    score: float