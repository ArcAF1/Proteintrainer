"""Embedding helper for ResearchMemory."""
from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

_default_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def default_embedding_fn(text: str) -> List[float]:
    return _default_model.encode(text, normalize_embeddings=True).tolist()
