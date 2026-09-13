from langfuse import Evaluation

from llm_eval.evaluators.judge import LLMJudge


def create_faithfulness_evaluator(judge: LLMJudge):
    def evaluator(
        *,
        input,
        output,
        **kwargs,
    ) -> Evaluation:
        contexts = "\n\n".join(
            output["contexts"]
        )

        prompt = f"""
Determine whether every factual claim in the answer is supported
by the retrieved context.

Question:
{input["question"]}

Retrieved context:
{contexts}

Answer:
{output["answer"]}

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