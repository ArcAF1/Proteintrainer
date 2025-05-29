"""Builds and saves the FAISS index.

Usage example:
    python src/indexer.py
"""
from pathlib import Path
import pickle


from typing import Iterable



import faiss
import numpy as np
from tqdm import tqdm

from .config import settings
from .embeddings import Embedder



def load_documents() -> list[str]:
    """Load text documents from the data directory."""
    docs = []
    for path in Path(settings.data_dir).rglob("*.txt"):
        docs.append(path.read_text())
    return docs


def build_index(docs: Iterable[str]) -> tuple[faiss.IndexFlatIP, list[str]]:
    embedder = Embedder()
    vectors = [embedder.encode(doc) for doc in tqdm(docs, desc="Embedding")]
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(np.stack(vectors))
    return index, list(docs)


def save_index(index: faiss.IndexFlatIP, docs: list[str]) -> None:

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = settings.index_dir / "pmc.faiss"
    faiss.write_index(index, str(index_path))
    with open(settings.index_dir / "pmc.pkl", "wb") as fh:
        pickle.dump(docs, fh)


def main() -> None:
    docs = load_documents()
    if not docs:
        print("No documents found in", settings.data_dir)
        return
    index, store = build_index(docs)
    save_index(index, store)
    print("Index built with", len(store), "documents")



if __name__ == "__main__":
    main()

