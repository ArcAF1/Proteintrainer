#!/usr/bin/env python3
"""Phase-1 data pipeline – download datasets, build full FAISS index, generate graph triplets.

This is a *thin* wrapper that orchestrates helpers inside `src/` so that
advanced users can run it from CLI or the GUI can call it in a thread.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable
import shutil

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))  # noqa: E402

from src import indexer, graph_builder, entity_extractor  # type: ignore
from src.config import settings  # type: ignore


ProgressCb = Callable[[float, str], None]


def _dummy_download(progress: ProgressCb | None = None) -> Iterable[str]:
    """Pretend to download a dataset – we just copy mini txt files multiple times."""
    sample_dir = Path(settings.data_dir) / "sample_txt"
    sample_dir.mkdir(parents=True, exist_ok=True)
    txt_path = sample_dir / "sample.txt"
    if not txt_path.exists():
        txt_path.write_text("Creatine improves muscle performance and may influence insulin signalling.")
    # replicate file 1k times to simulate corpus
    docs = []
    for i in range(1000):
        dst = sample_dir / f"doc_{i}.txt"
        shutil.copy(txt_path, dst)
        docs.append(dst.read_text())
        if progress and i % 100 == 0:
            progress(i / 1000, f"Downloading mock corpus … {i}/1000")
    if progress:
        progress(1.0, "Datasets downloaded (mock)")
    return docs


def run(progress: ProgressCb | None = None) -> None:
    # Step-1 Download
    if progress:
        progress(0.01, "Downloading datasets …")
    docs = _dummy_download(progress)

    # Step-2 Index
    if progress:
        progress(0.2, "Building full FAISS index …")
    index, store = indexer.build_full_index(docs)
    indexer.save_index(index, store)  # re-use helper
    if progress:
        progress(0.6, f"Index built with {len(store)} docs")

    # Step-3 Graph population
    if progress:
        progress(0.65, "Extracting entities …")
    articles, relations = entity_extractor.extract_triplets_bulk(docs, progress)
    if progress:
        progress(0.8, "Ingesting into Neo4j …")
    graph_builder.build_graph(articles, relations)
    if progress:
        progress(1.0, "✅ Data pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run() 