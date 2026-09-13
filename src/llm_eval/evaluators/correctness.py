from langfuse import Evaluation

from llm_eval.evaluators.judge import LLMJudge


def create_correctness_evaluator(judge: LLMJudge):
    def evaluator(
        *,
        input,
        output,
        expected_output,
        **kwargs,
    ) -> Evaluation:
        prompt = f"""
Evaluate whether the candidate answer is factually consistent
with the reference answer.

Question:
{input["question"]}

Reference answer:
{expected_output["answer"]}

Candidate answer:
{output["answer"]}

Return a score from 0 to 1.

1.0 = fully correct
0.5 = partially correct
0.0 = incorrect
""".strip()

        result = judge.evaluate(prompt)

        return Evaluation(
            name="correctness",
            value=result.score,
            comment=result.reason,
        )

    return evaluator