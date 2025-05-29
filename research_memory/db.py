"""Core logic for research memory."""
from __future__ import annotations

from array import array
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import numpy as np
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Entry, Embedding


class MemoryConfig(BaseModel):
    db_path: Path = Path.home() / "research_memory" / "research.db"
    entries_dir: Path = Path.home() / "research_memory" / "entries"


class ResearchMemory:
    """Handles storage and retrieval of notebook entries."""

    def __init__(self, embedding_fn: Callable[[str], List[float]], config: MemoryConfig | None = None) -> None:
        self.embedding_fn = embedding_fn
        self.config = config or MemoryConfig()
        self.config.entries_dir.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.config.db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _store_file(self, entry: Entry) -> None:
        import yaml

        path = self.config.entries_dir / f"{entry.id}.md"
        front_matter = {
            "id": entry.id,
            "type": entry.type,
            "title": entry.title,
            "authors": [],
            "date": entry.created_at.isoformat(),
            "status": entry.status,
            "confidence": entry.confidence,
            "tags": entry.tags.split(",") if entry.tags else [],
            "links": entry.links,
        }
        content = f"---\n{yaml.safe_dump(front_matter)}---\n{entry.body_md}\n"
        path.write_text(content)

    def add_entry(
        self,
        *,
        type: str,
        title: str,
        body: str,
        status: str = "open",
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        links: Optional[List[dict[str, Any]]] = None,
        revises_id: Optional[str] = None,
    ) -> Entry:
        eid = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        vector = array("f", self.embedding_fn(body)).tobytes()
        with self.Session() as session:
            entry = Entry(
                id=eid,
                type=type,
                title=title,
                body_md=body,
                status=status,
                confidence=confidence,
                tags=",".join(tags) if tags else None,
                links=links,
                revises_id=revises_id,
            )
            session.add(entry)
            session.add(Embedding(entry_id=eid, vector=vector))
            session.commit()
            self._store_file(entry)
            return entry

    def update_status(self, entry_id: str, status: str, confidence: Optional[float] = None) -> Entry:
        with self.Session() as session:
            entry = session.get(Entry, entry_id)
            if not entry:
                raise ValueError("Entry not found")
            new_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            new_entry = Entry(
                id=new_id,
                type=entry.type,
                title=entry.title,
                body_md=entry.body_md,
                status=status,
                confidence=confidence if confidence is not None else entry.confidence,
                tags=entry.tags,
                links=entry.links,
                revises_id=entry.id,
            )
            session.add(new_entry)
            session.add(
                Embedding(entry_id=new_id, vector=session.get(Embedding, entry_id).vector)
            )
            session.commit()
            self._store_file(new_entry)
            return new_entry

    def _all_embeddings(self, session: Session) -> tuple[np.ndarray, list[Entry]]:
        rows = session.execute(select(Embedding, Entry).join(Entry)).all()
        vectors = []
        entries = []
        for emb, entry in rows:
            vec = np.frombuffer(emb.vector, dtype="float32")
            vectors.append(vec)
            entries.append(entry)
        if not vectors:
            return np.empty((0, 0), dtype="float32"), []
        matrix = np.stack(vectors)
        return matrix, entries

    def search(self, text: str, k: int = 5) -> List[tuple[Entry, float]]:
        query_vec = np.array(self.embedding_fn(text), dtype="float32")
        query_vec /= np.linalg.norm(query_vec) + 1e-9
        with self.Session() as session:
            matrix, entries = self._all_embeddings(session)
            if matrix.size == 0:
                return []
            matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
            scores = matrix_norm @ query_vec
            idx = np.argsort(scores)[-k:][::-1]
            return [(entries[i], float(scores[i])) for i in idx]

    def list_entries(
        self,
        type: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Entry]:
        with self.Session() as session:
            stmt = select(Entry)
            if type:
                stmt = stmt.where(Entry.type == type)
            if status:
                stmt = stmt.where(Entry.status == status)
            if tag:
                stmt = stmt.where(Entry.tags.like(f"%{tag}%"))
            return list(session.scalars(stmt))


Entry = Entry  # re-export for __all__
