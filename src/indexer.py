"""Builds and saves the FAISS index.
Usage example:
    python src/indexer.py
"""
from pathlib import Path
import pickle
from typing import Iterable, Generator, Tuple, Dict

import faiss
import numpy as np
from tqdm import tqdm

from .config import settings
from .embeddings import Embedder


def iter_documents() -> Generator[Tuple[str, Dict[str, str]], None, None]:
    """Yield document text with metadata."""
    for path in Path(settings.data_dir).rglob("*.txt"):
        text = path.read_text()
        meta = {"source": path.parent.name, "path": str(path)}
        yield text, meta


def build_index(doc_iter: Iterable[Tuple[str, Dict[str, str]]]) -> Tuple[faiss.IndexFlatIP, list[str], list[Dict[str, str]]]:
    embedder = Embedder()
    index = None
    docs: list[str] = []
    metas: list[Dict[str, str]] = []
    batch_texts: list[str] = []
    batch_meta: list[Dict[str, str]] = []
    for text, meta in doc_iter:
        batch_texts.append(text)
        batch_meta.append(meta)
        if len(batch_texts) >= 32:
            vecs = embedder.encode_batch(batch_texts)
            faiss.normalize_L2(vecs)
            if index is None:
                dim = vecs.shape[1]
                index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            docs.extend(batch_texts)
            metas.extend(batch_meta)
            batch_texts, batch_meta = [], []
    if batch_texts:
        vecs = embedder.encode_batch(batch_texts)
        faiss.normalize_L2(vecs)
        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        docs.extend(batch_texts)
        metas.extend(batch_meta)
    return index, docs, metas


def save_index(index: faiss.IndexFlatIP, docs: list[str], metas: list[Dict[str, str]]) -> None:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = settings.index_dir / "pmc.faiss"
    faiss.write_index(index, str(index_path))
    with open(settings.index_dir / "pmc.pkl", "wb") as fh:
        pickle.dump({"docs": docs, "meta": metas}, fh)


def main() -> None:
    docs_iter = iter_documents()
    index, docs, metas = build_index(docs_iter)
    if not docs:
        print("No documents found in", settings.data_dir)
        return
    save_index(index, docs, metas)
    print("Index built with", len(docs), "documents")


if __name__ == "__main__":
    main()
