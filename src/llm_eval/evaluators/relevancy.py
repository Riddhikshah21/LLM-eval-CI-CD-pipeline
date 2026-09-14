from langfuse import Evaluation

from llm_eval.evaluators.judge import LLMJudge


def create_relevancy_evaluator(
    judge: LLMJudge,
):
    def evaluator(
        *,
        input,
        output,
        **kwargs,
    ) -> Evaluation:
        prompt = f"""
Evaluate how relevant the response is to the user's input.

Input:
{input}

Response:
{output["response"]}

Return a score from 0 to 1.

1.0 = directly relevant
0.5 = partially relevant
0.0 = irrelevant
""".strip()

        result = judge.evaluate(prompt)

        return Evaluation(
            name="answer_relevancy",
            value=result.score,
            comment=result.reason,
        )

    return evaluator