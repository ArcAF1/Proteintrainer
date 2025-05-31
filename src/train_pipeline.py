from __future__ import annotations

"""End-to-end training / ingestion pipeline.

Runs the three core steps in order:
1. Download & extract datasets (data_ingestion.py)
2. Build / refresh FAISS index (indexer.py)
3. Populate Neo4j graph (graph_builder.py)

The function is kept intentionally lightweight so that it can be invoked from
both the GUI button and CLI:

    python -m src.train_pipeline
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Callable
from collections import defaultdict
import logging, sys

from . import data_ingestion, indexer, graph_builder, relation_extraction
from .config import settings

# Progress callback type
ProgressCallback = Optional[Callable[[float, str], None]]

log_fh = open("training.log", "a", buffering=1)

def _tee(msg: str):
    print(msg, file=log_fh)
    print(msg)


def _load_graph_inputs() -> tuple[List[dict], List[dict]]:
    """Attempt to load extracted article + relation JSON stubs if present.

    The current repo does NOT yet include an NLP entity extractor – therefore we
    look for pre-parsed helper files created by the user (or tests). If they do
    not exist we return empty lists so that the call to graph_builder is still
    valid.
    """
    articles_path = Path("data/parsed/articles.json")
    relations_path = Path("data/parsed/relations.json")
    articles: List[dict] = []
    relations: List[dict] = []
    if articles_path.exists():
        articles = json.loads(articles_path.read_text())
    if relations_path.exists():
        relations = json.loads(relations_path.read_text())
    return articles, relations


def _extract_from_text_files(progress_callback: ProgressCallback = None) -> tuple[List[dict], List[dict]]:
    """Run fast relation extraction over recent txt docs (heuristic)."""
    articles: List[dict] = []
    relations: List[dict] = []
    
    text_files = list(Path(settings.data_dir).rglob("*.txt"))
    total_files = len(text_files)
    
    for idx, path in enumerate(text_files):
        if progress_callback:
            progress = idx / total_files
            progress_callback(progress, f"Extracting relations from {path.name} ({idx+1}/{total_files})")
        
        text = path.read_text(encoding="utf-8", errors="ignore")
        ents, rels = relation_extraction.extract(text[:5000])  # limit length for speed
        if ents:
            art_id = path.stem
            articles.append({"id": art_id, "title": path.stem, "entities": ents})
            for r in rels:
                r["article_id"] = art_id
            relations.extend(rels)
    
    if progress_callback:
        progress_callback(1.0, f"Extracted {len(relations)} relations from {total_files} files")
    
    return articles, relations


def main(progress_callback: ProgressCallback = None) -> None:
    # Overall pipeline has 3 major steps
    def step_progress(step: int, progress: float, message: str):
        if progress_callback:
            # Each step gets 1/3 of the total progress
            overall_progress = (step - 1) / 3 + (progress / 3)
            progress_callback(overall_progress, message)
    
    print("\n[train_pipeline] Step 1/3 – downloading datasets…")
    print("This step will download biomedical datasets with 100GB available space.")
    print("The system will automatically select datasets based on available space.")
    print("Larger datasets like PubMed Baseline will be downloaded in full.\n")
    
    try:
        data_ingestion.main(
            progress_callback=lambda p, m: step_progress(1, p, m) if progress_callback else None,
            available_gb=100.0  # Use full 100GB
        )
    except Exception as e:
        print(f"\n⚠️  Dataset download issue: {str(e)}")
        print("Continuing with available data...\n")

    print("\n[train_pipeline] Step 2/3 – building FAISS index…")
    if progress_callback:
        step_progress(2, 0, "Building FAISS index...")
    try:
        indexer.main()
        if progress_callback:
            step_progress(2, 1, "FAISS index built")
    except Exception as e:
        print(f"\n⚠️  Indexing issue: {str(e)}")
        print("You may need to download data first\n")
        if progress_callback:
            step_progress(2, 1, f"Indexing failed: {str(e)}")

    print("\n[train_pipeline] Step 3/3 – populating Neo4j graph…")
    articles, relations = _load_graph_inputs()
    if not articles:
        print("[train_pipeline] No parsed JSON entities found – performing quick extraction …")
        articles, relations = _extract_from_text_files(
            progress_callback=lambda p, m: step_progress(3, p * 0.5, m) if progress_callback else None
        )

    try:
        if progress_callback:
            step_progress(3, 0.5, "Populating Neo4j graph...")
        graph_builder.build_graph(articles, relations)
        print(f"[train_pipeline] Inserted {len(articles)} articles & {len(relations)} relations into Neo4j.")
        if progress_callback:
            step_progress(3, 1, f"Graph populated with {len(articles)} articles & {len(relations)} relations")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n⚠️  Graph build issue: {exc}")
        print("The system will still work, but with limited knowledge graph capabilities.")
        if progress_callback:
            step_progress(3, 1, f"Graph build skipped: {exc}")

    print("\n[train_pipeline] ✅ Pipeline finished.")
    print("\nNext steps:")
    print("1. Run ./start.command to open the GUI")
    print("2. The system will fetch recent papers via API on-demand")
    print("3. Additional datasets can be downloaded as needed")
    print("\nThe system is configured to use up to 100GB of space for datasets.")
    
    if progress_callback:
        progress_callback(1.0, "✅ Pipeline complete!")


if __name__ == "__main__":
    main() 