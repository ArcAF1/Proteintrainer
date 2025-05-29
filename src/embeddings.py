"""Sentence-Transformers embedding wrapper.

Usage example:
    from embeddings import Embedder
    vec = Embedder().encode("hello")
"""
from typing import Optional, Iterable
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

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.array(vectors, dtype="float32")
