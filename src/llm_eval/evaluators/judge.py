from openai import OpenAI
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    score: float = Field(ge=0, le=1)
    reason: str


class LLMJudge:
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model

    def evaluate(self, prompt: str) -> JudgeResult:
        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=JudgeResult,
        )

        return response.output_parsed