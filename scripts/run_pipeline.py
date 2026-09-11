from llm_eval.pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()

    result = pipeline.run(
        "How long do I have to return an item?"
    )

    print("Answer:", result.answer)
    print("Sources:", result.retrieved_sources)
    print("Latency:", result.latency_ms)
    print("Input tokens:", result.input_tokens)
    print("Output tokens:", result.output_tokens)


if __name__ == "__main__":
    main()