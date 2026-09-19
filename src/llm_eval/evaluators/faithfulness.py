from langfuse import Evaluation

from llm_eval.evaluators.judge import LLMJudge

ABSTENTION_TEXT = "I don't have enough information to answer that question."


def create_faithfulness_evaluator(
    judge: LLMJudge,
):
    def evaluator(
        *,
        input,
        output,
        metadata,
        **kwargs,
    ) -> Evaluation:
        response = output["response"]

        # An expected abstention contains no unsupported factual claim.
        if metadata.get("should_abstain", False):
            abstained = ABSTENTION_TEXT.lower() in response.lower()

            return Evaluation(
                name="faithfulness",
                value=1.0 if abstained else 0.0,
                comment=(
                    "Correctly abstained."
                    if abstained
                    else "Expected abstention but model answered."
                ),
            )

        contexts = output.get("metadata", {}).get("contexts", [])

        context_text = "\n\n".join(contexts)

        prompt = f"""
Evaluate whether every factual claim in the response is
supported by the retrieved context.

Question:
{input["question"]}

Retrieved context:
{context_text}

Response:
{response}

Scoring:
1.0 = every factual claim is explicitly supported
0.5 = some claims are supported but others are unsupported
0.0 = the response contradicts the context or invents facts

Do not penalize concise wording or paraphrasing.
Judge factual support only.
""".strip()

        result = judge.evaluate(prompt)

        return Evaluation(
            name="faithfulness",
            value=result.score,
            comment=result.reason,
        )

    return evaluator
