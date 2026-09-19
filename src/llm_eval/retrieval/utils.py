import hashlib
import re
from pathlib import Path

from llm_eval.schemas import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _document_id(source: str, content: str) -> str:
    value = f"{source}:{content}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_documents(
    knowledge_base_path: str | Path,
) -> list[Document]:
    path = Path(knowledge_base_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base path does not exist: {path}")

    files = sorted(path.glob("*.md"))

    if not files:
        raise ValueError(f"No Markdown documents found in {path}")

    documents: list[Document] = []

    for file in files:
        content = file.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            Document(
                id=_document_id(file.name, content),
                source=file.name,
                content=content,
            )
        )

    if not documents:
        raise ValueError("Knowledge base contains no usable documents")

    return documents
