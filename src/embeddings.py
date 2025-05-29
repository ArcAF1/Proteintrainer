"""Sentence-Transformers embedding wrapper.

Usage example:
    from embeddings import Embedder
    vec = Embedder().encode("hello")
"""


from typing import Optional


import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


class Embedder:
    """Lightweight embedding helper."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = SentenceTransformer(model_name or settings.embed_model)

    def encode(self, text: str) -> np.ndarray:
        vector = self.model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype="float32")


