import os

from langfuse import get_client

from llm_eval.evaluators.abstention import abstention_evaluator
from llm_eval.pipeline import RAGPipeline
from llm_eval.sync_dataset import DATASET_NAME


def run_experiment() -> None:
    langfuse = get_client()

    dataset = langfuse.get_dataset(DATASET_NAME)

    pipeline = RAGPipeline()

    def task(*, item, **kwargs):
        question = item.input["question"]

        result = pipeline.run(question)

        return {
            "answer": result.answer,
            "sources": result.retrieved_sources,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    result = dataset.run_experiment(
        name=f"eval-{os.getenv('GITHUB_SHA', 'local')}",
        description="LLM/RAG regression evaluation",
        task=task,
        evaluators=[
            abstention_evaluator,
        ],
        metadata={
            "git_sha": os.getenv("GITHUB_SHA", "local"),
            "branch": os.getenv("GITHUB_REF_NAME", "local"),
        },
        max_concurrency=3,
    )

    print(result.format())

    langfuse.flush()


if __name__ == "__main__":
    run_experiment()    