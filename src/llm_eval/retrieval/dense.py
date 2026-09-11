from typing import Protocol

import faiss
import numpy as np
from openai import OpenAI

from llm_eval.schemas import Document, RetrievedDocument


class EmbeddingProvider(Protocol):
    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        ...

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        ...


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str,
    ) -> None:
        self.client = OpenAI()
        self.model = model

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        vectors = np.asarray(
            [
                item.embedding
                for item in response.data
            ],
            dtype=np.float32,
        )

        return vectors

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )

        return np.asarray(
            [response.data[0].embedding],
            dtype=np.float32,
        )


class DenseRetriever:
    def __init__(
        self,
        documents: list[Document],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        if not documents:
            raise ValueError("Documents cannot be empty")

        self.documents = documents
        self.embedding_provider = embedding_provider

        embeddings = self.embedding_provider.embed_documents(
            [document.content for document in documents]
        )

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = (
            self.embedding_provider.embed_query(query)
        )

        faiss.normalize_L2(query_embedding)

        k = min(top_k, len(self.documents))

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results: list[RetrievedDocument] = []

        for score, index in zip(
            scores[0],
            indices[0],
            strict=True,
        ):
            if index < 0:
                continue

            document = self.documents[int(index)]

            results.append(
                RetrievedDocument(
                    id=document.id,
                    source=document.source,
                    content=document.content,
                    score=float(score),
                )
            )

        return results