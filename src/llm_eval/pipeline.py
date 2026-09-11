from pathlib import Path

from llm_eval.config import load_model_config
from llm_eval.llm_client import OpenAIClient
from llm_eval.schemas import PipelineResult
from llm_eval.retrieval.factory import create_hybrid_retriever

class RAGPipeline:
    def __init__(self) -> None:
        config = load_model_config()

        self.top_k = config.rag.top_k

        self.retriever = create_hybrid_retriever(
            config=config.rag,
            knowledge_base_path="knowledge_base",
        )

        self.llm = OpenAIClient(
            model=config.model.name,
            temperature=config.model.temperature,
        )

        self.system_prompt = Path(
            "prompts/system_prompt.txt"
        ).read_text(encoding="utf-8")

    def run(self, question: str) -> PipelineResult:
        documents = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
        )

        context = "\n\n".join(
            f"Source: {doc.source}\n{doc.content}"
            for doc in documents
        )

        user_prompt = f"""
Context:
{context}

Question:
{question}
""".strip()

        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )

        return PipelineResult(
            question=question,
            answer=response.text,
            retrieved_context=[
                doc.content for doc in documents
            ],
            retrieved_sources=[
                doc.source for doc in documents
            ],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )