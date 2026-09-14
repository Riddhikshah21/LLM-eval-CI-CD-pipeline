from pathlib import Path
from time import perf_counter
from typing import Any

from langfuse import observe

from llm_eval.config import load_model_config
from llm_eval.cost import calculate_cost
from llm_eval.llm_client import OpenAIClient
from llm_eval.retrieval.factory import create_hybrid_retriever
from llm_eval.systems.base import SystemResult


class RAGSystem:
    def __init__(self) -> None:
        config = load_model_config()

        self.top_k = config.rag.top_k

        self.retriever = create_hybrid_retriever(
            config=config.rag,
            knowledge_base_path="knowledge_base",
        )
        self.pricing = config.model.pricing
        self.llm = OpenAIClient(
            model=config.model.name,
            temperature=config.model.temperature,
        )

        self.system_prompt = Path("prompts/system_prompt.txt").read_text(
            encoding="utf-8"
        )

    @observe(name="rag-system")
    def run(
        self,
        input_data: dict[str, Any],
    ) -> SystemResult:
        question = input_data["question"]

        start = perf_counter()

        documents = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
        )

        context = "\n\n".join(
            f"Source: {document.source}\n{document.content}" for document in documents
        )

        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )

        total_latency_ms = (perf_counter() - start) * 1000
        cost_usd = calculate_cost(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            input_per_1m_tokens_usd=(self.pricing.input_per_1m_tokens_usd),
            output_per_1m_tokens_usd=(self.pricing.output_per_1m_tokens_usd),
        )
        return SystemResult(
            response=response.text,
            latency_ms=total_latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost_usd,
            metadata={
                "contexts": [document.content for document in documents],
                "sources": [document.source for document in documents],
                "application_type": "rag",
            },
        )
