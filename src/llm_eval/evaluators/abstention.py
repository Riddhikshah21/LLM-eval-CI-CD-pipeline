from langfuse import Evaluation


ABSTENTION_TEXT = "I don't have enough information to answer that question."


def abstention_evaluator(
    *,
    output,
    metadata,
    **kwargs,
) -> Evaluation:
    should_abstain = metadata.get("should_abstain", False)

    answer = output["answer"]

    abstained = ABSTENTION_TEXT.lower() in answer.lower()

    passed = (
        abstained
        if should_abstain
        else not abstained
    )

    return Evaluation(
        name="abstention_accuracy",
        value=1.0 if passed else 0.0,
    )