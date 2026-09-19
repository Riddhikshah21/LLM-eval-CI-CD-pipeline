import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client

from llm_eval.config import (
    load_evaluation_config,
    load_model_config,
)
from llm_eval.evaluators.abstention import abstention_evaluator
from llm_eval.evaluators.correctness import create_correctness_evaluator
from llm_eval.evaluators.faithfulness import create_faithfulness_evaluator
from llm_eval.evaluators.judge import LLMJudge
from llm_eval.evaluators.relevancy import create_relevancy_evaluator
from llm_eval.metrics import aggregate_metrics
from llm_eval.sync_dataset import DATASET_NAME
from llm_eval.systems import create_system

REPORT_PATH = Path("reports/evaluation.json")


load_dotenv()

def run_experiment() -> None:
    langfuse = get_client()

    if not langfuse.auth_check():
        raise RuntimeError(
            "Langfuse authentication failed. "
            "Check LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL."
        )

    model_config = load_model_config()
    evaluation_config = load_evaluation_config()

    dataset = langfuse.get_dataset(DATASET_NAME)

    system = create_system(model_config.application.type)

    judge = LLMJudge(model=evaluation_config.judge.model)

    git_sha = os.getenv(
        "GITHUB_SHA",
        "local",
    )

    branch = os.getenv(
        "GITHUB_REF_NAME",
        "local",
    )

    experiment_name = f"eval-{branch}-{git_sha[:8]}"

    def task(*, item, **kwargs) -> dict:
        result = system.run(item.input)

        return {
            "response": result.response,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "metadata": result.metadata,
        }

    try:
        result = dataset.run_experiment(
            name=experiment_name,
            description=("AI application regression evaluation"),
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
                "application_type": (model_config.application.type),
                "model": model_config.model.name,
                "judge_model": (evaluation_config.judge.model),
            },
            max_concurrency=3,
        )

        metrics = aggregate_metrics(
            item_results=result.item_results,
            faithfulness_threshold=(evaluation_config.thresholds.faithfulness),
        )

        report = {
            "experiment": {
                "name": experiment_name,
                "git_sha": git_sha,
                "branch": branch,
                "application_type": (model_config.application.type),
                "model": model_config.model.name,
                "judge_model": (evaluation_config.judge.model),
                "langfuse_url": (result.dataset_run_url),
            },
            "metrics": metrics,
        }

        REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        REPORT_PATH.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        if result.dataset_run_url:
            print(f"\nLangfuse experiment: {result.dataset_run_url}")

    finally:
        langfuse.flush()


if __name__ == "__main__":
    run_experiment()
