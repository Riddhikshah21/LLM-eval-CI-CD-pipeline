import time

from langfuse import observe
from openai import OpenAI

from llm_eval.schemas import LLMResponse


class OpenAIClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    @observe(name="llm-generation", as_type="generation")
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )
