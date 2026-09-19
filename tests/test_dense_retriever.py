from typing import ClassVar

import numpy as np


class FakeEmbeddingProvider:
    vectors: ClassVar[dict[str, list[float]]] = {
        "refund policy return item": [1.0, 0.0, 0.0],
        "shipping delivery package": [0.0, 1.0, 0.0],
        "password account reset": [0.0, 0.0, 1.0],
        "return refund": [1.0, 0.0, 0.0],
    }

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return np.asarray(
            [self.vectors[text] for text in texts],
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        return np.asarray(
            [self.vectors[text]],
            dtype=np.float32,
        )