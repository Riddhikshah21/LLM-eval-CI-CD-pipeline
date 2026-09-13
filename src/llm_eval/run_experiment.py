import os

from langfuse import get_client

from llm_eval.config import load_evaluation_config, load_model_config
from llm_eval.evaluators.abstention import abstention_evaluator
from llm_eval.evaluators.correctness import create_correctness_evaluator
from llm_eval.evaluators.faithfulness import create_faithfulness_evaluator
from llm_eval.evaluators.judge import LLMJudge
from llm_eval.evaluators.relevancy import create_relevancy_evaluator
from llm_eval.pipeline import RAGPipeline
from llm_eval.sync_dataset import DATASET_NAME


def run_experiment() -> None:
    langfuse = get_client()

    model_config = load_model_config()
    evaluation_config = load_evaluation_config()

    dataset = langfuse.get_dataset(
        DATASET_NAME
    )

    pipeline = RAGPipeline()

    judge = LLMJudge(
        model=evaluation_config.judge.model
    )

    def task(*, item, **kwargs):
        question = item.input["question"]

        result = pipeline.run(question)

        return {
            "answer": result.answer,
            "contexts": result.retrieved_context,
            "sources": result.retrieved_sources,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    git_sha = os.getenv(
        "GITHUB_SHA",
        "local",
    )

    branch = os.getenv(
        "GITHUB_REF_NAME",
        "local",
    )

    experiment_name = (
        f"eval-{branch}-{git_sha[:8]}"
    )

    try:
        result = dataset.run_experiment(
            name=experiment_name,
            description="LLM/RAG regression evaluation",
            task=task,
            evaluators=[
                abstention_evaluator,
                create_correctness_evaluator(judge),
                create_relevancy_evaluator(judge),
                create_faithfulness_evaluator(judge),
            ],
            metadata={
                "git_sha": git_sha,
                "branch": branch,
                "model": model_config.model.name,
                "judge_model": evaluation_config.judge.model,
            },
            max_concurrency=3,
        )

        print(result.format())

        if result.dataset_run_url:
            print(
                f"\nLangfuse experiment: "
                f"{result.dataset_run_url}"
            )

    finally:
        langfuse.flush()


if __name__ == "__main__":
    run_experiment()