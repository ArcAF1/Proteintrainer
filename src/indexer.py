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


# NEW: quick demo/mini index builder ------------------------------------------------

def build_mini_index(max_docs: int = 1000) -> None:
    """Build a *small* FAISS index so that first-time users can chat immediately.

    Downloads a sample of PubMed abstracts via the NIH E-utilities API, embeds
    them and writes `indexes/mini.faiss` + `indexes/mini.pkl`.
    """
    import requests, textwrap, time

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    mini_index_path = settings.index_dir / "mini.faiss"
    mini_store_path = settings.index_dir / "mini.pkl"

    # Skip if already exists
    if mini_index_path.exists() and mini_store_path.exists():
        print("[mini-index] Existing mini index found – skipping build.")
        return

    print(f"[mini-index] Fetching {max_docs} sample PubMed abstracts …")
    
    api_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        "?db=pubmed&term=biomedical[Title]&retmax=" + str(max_docs) + "&retmode=json"
    )
    ids = requests.get(api_url, timeout=30).json()["esearchresult"]["idlist"]

    # Fetch abstracts in batches of 200
    abstracts: list[str] = []
    for i in range(0, len(ids), 200):
        chunk = ",".join(ids[i : i + 200])
        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=pubmed&id=" + chunk + "&rettype=abstract&retmode=text"
        )
        txt = requests.get(fetch_url, timeout=30).text
        # crude split – abstracts separated by blank lines
        abstracts += [a.strip() for a in txt.split("\n\n") if len(a.strip()) > 50]
        time.sleep(0.34)  # stay well below NCBI rate limit
        if len(abstracts) >= max_docs:
            abstracts = abstracts[:max_docs]
            break

    if not abstracts:
        raise RuntimeError("Failed to download sample abstracts for mini index.")

    print(f"[mini-index] Embedding {len(abstracts)} abstracts …")
    embedder = Embedder()
    vecs = [embedder.encode(a) for a in abstracts]
    dim = vecs[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(np.stack(vecs))

    faiss.write_index(index, str(mini_index_path))
    with open(mini_store_path, "wb") as fh:
        pickle.dump(abstracts, fh)
    print("[mini-index] ✅ mini index built and saved.")

# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full index builder (IVF256 + Flat) for larger corpora


def build_full_index(docs: Iterable[str]) -> tuple[faiss.Index, list[str]]:  # noqa: D401
    """Build a FAISS IVF256 index from an iterable of documents."""
    embedder = Embedder()
    vectors = [embedder.encode(doc) for doc in tqdm(docs, desc="Embedding (full)")]

    dim = vectors[0].shape[0]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 256, faiss.METRIC_INNER_PRODUCT)
    vecmat = np.stack(vectors)
    index.train(vecmat)
    index.add(vecmat)
    return index, list(docs)


if __name__ == "__main__":
    main()

