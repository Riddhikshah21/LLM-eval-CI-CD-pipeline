from llm_eval.systems.base import SystemUnderTest
from llm_eval.systems.rag import RAGSystem


def create_system(
    application_type: str,
) -> SystemUnderTest:
    if application_type == "rag":
        return RAGSystem()

    raise ValueError(f"Unsupported application type: {application_type}")
