 # src/embeddings/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of texts into normalized embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return np.array(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert one user query into a normalized embedding vector.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.array(embedding, dtype="float32")
