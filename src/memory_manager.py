from __future__ import annotations
"""Thin wrapper around research_memory.ResearchMemory used by the GUI."""

from pathlib import Path
from typing import List
import structlog
import numpy as np

from research_memory import ResearchMemory
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

_DB_PATH = Path("data/memory/research.db")
_ENTRIES_DIR = Path("data/memory_entries")

_mem: ResearchMemory | None = None
_model: SentenceTransformer | None = None


def get_memory() -> ResearchMemory:
    """Get or create the global memory instance."""
    global _mem, _model
    
    if _mem is None:
        # Use CPU for memory manager to avoid MPS memory conflicts
        _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Define embedding function
        def embed_fn(text: str) -> list[float]:
            embedding = _model.encode(text)
            return embedding.tolist()
        
        # Set up paths
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize ResearchMemory with only the expected parameters
        _mem = ResearchMemory(embedding_fn=embed_fn, db_path=str(_DB_PATH))
        logger.info("Initialized research memory", db_path=str(_DB_PATH))
    return _mem


def save_finding(question: str, answer: str) -> None:
    """Persist a Q/A pair as a note entry."""
    mem = get_memory()
    mem.add_entry(type="qa", title=question[:80], body=answer)


def recall(query: str, k: int = 3) -> List[str]:
    mem = get_memory()
    hits = mem.search(query, k=k)
    return [h.entry.body for h in hits] 