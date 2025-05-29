"""RAG pipeline with Ollama-based LLM.

Usage example:
    from rag_chat import answer
    import asyncio
    print(asyncio.run(answer("Vad är aspirin?")))
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
import requests

from .config import settings
from .embeddings import Embedder

SYSTEM_PROMPT = (
    "Du är en medicinsk forskningsassistent. "
    "Använd endast informationen i källorna och ange (Dok n) som referens. "
    "Svaren är inte medicinska råd."
)


class RAGChat:
    """Simple retrieval-augmented chat."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        index_path = settings.index_dir / "pmc.faiss"
        store_path = settings.index_dir / "pmc.pkl"
        if not index_path.exists() or not store_path.exists():
            raise FileNotFoundError("Index files missing. Run indexer.py first.")
        self.index = faiss.read_index(str(index_path))
        with open(store_path, "rb") as fh:
            self.docs = pickle.load(fh)

    def retrieve(self, query: str) -> List[str]:
        vector = self.embedder.encode(query).reshape(1, -1)
        scores, ids = self.index.search(vector, settings.top_k)
        return [self.docs[i] for i in ids[0]]

    async def generate(self, prompt: str) -> str:
        payload = {"model": settings.llm_model, "prompt": prompt}

        def call() -> str:
            resp = requests.post(f"{settings.ollama_host}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, call)

    async def answer(self, question: str) -> str:
        docs = self.retrieve(question)
        context = "\n---\n".join(f"(Dok {i+1}) {d}" for i, d in enumerate(docs))
        prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nFråga: {question}\nSvar:"
        return await self.generate(prompt)


_chat: RAGChat | None = None


def get_chat() -> RAGChat:
    global _chat
    if _chat is None:
        _chat = RAGChat()
    return _chat


async def answer(question: str) -> str:
    return await get_chat().answer(question)
