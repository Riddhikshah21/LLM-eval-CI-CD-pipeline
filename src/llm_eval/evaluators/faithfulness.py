from langfuse import Evaluation

from llm_eval.evaluators.judge import LLMJudge


def create_faithfulness_evaluator(
    judge: LLMJudge,
):
    def evaluator(
        *,
        input,
        output,
        **kwargs,
    ) -> Evaluation:
        contexts = output.get(
            "metadata",
            {},
        ).get(
            "contexts",
            [],
        )

        context_text = "\n\n".join(contexts)

        prompt = f"""
Determine whether every factual claim in the response is
supported by the retrieved context.

Input:
{input}

Retrieved context:
{context_text}

Response:
{output["response"]}

Return a score from 0 to 1.

1.0 = all claims supported
0.5 = partially supported
0.0 = unsupported or contradictory
""".strip()

        result = judge.evaluate(prompt)

        return Evaluation(
            name="faithfulness",
            value=result.score,
            comment=result.reason,
        )

    return evaluator