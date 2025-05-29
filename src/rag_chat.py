"""RAG pipeline functions.

Usage example:
    from rag_chat import answer
    response = asyncio.run(answer("What is aspirin?"))
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import List

try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    faiss = None
import numpy as np
from ctransformers import AutoModelForCausalLM

from .config import settings
from .embeddings import Embedder


class RAGChat:
    def __init__(self) -> None:
        self.embedder = Embedder()
        index_path = settings.index_dir / "pmc.faiss"
        store_path = settings.index_dir / "pmc.pkl"
        if not index_path.exists() or not store_path.exists():
            raise FileNotFoundError("Index files not found. Run indexer first.")
        if faiss is None:
            raise ImportError("faiss is required for retrieval but not installed")
        self.index = faiss.read_index(str(index_path))
        with open(store_path, "rb") as fh:
            self.docs = pickle.load(fh)
        model_path = settings.model_dir / settings.llm_model
        if not model_path.exists():
            raise FileNotFoundError(f"LLM weights missing: {model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
        )

    def retrieve(self, query: str) -> List[str]:
        vector = self.embedder.encode(query).reshape(1, -1)
        scores, ids = self.index.search(vector, settings.top_k)
        return [self.docs[i] for i in ids[0]]

    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.llm, prompt)

    async def answer(self, question: str) -> str:
        docs = self.retrieve(question)
        context = "\n---\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return await self.generate(prompt)


_chat = None


def get_chat() -> RAGChat:
    global _chat
    if _chat is None:
        _chat = RAGChat()
    return _chat


async def answer(question: str) -> str:
    return await get_chat().answer(question)

