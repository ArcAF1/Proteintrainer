from __future__ import annotations

import json
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
from sqlalchemy import (
    JSON,
    BLOB,
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import Session, declarative_base

from .models import Entry, SearchHit

DB_DIR = Path.home() / "research_memory"
DB_PATH = DB_DIR / "research.db"
ENTRIES_DIR = DB_DIR / "entries"

Base = declarative_base()


class EntryRow(Base):
    __tablename__ = "entry"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime)
    type = Column(String)
    title = Column(String)
    body_md = Column(Text)
    status = Column(String)
    confidence = Column(Float)
    tags = Column(String)
    links = Column(JSON)
    revises_id = Column(String, ForeignKey("entry.id"))


class EmbeddingRow(Base):
    __tablename__ = "embedding"
    entry_id = Column(String, ForeignKey("entry.id"), primary_key=True)
    vector = Column(BLOB)


class ResearchMemory:
    """Electronic lab notebook backend."""

    def __init__(self, embedding_fn: Callable[[str], List[float]], path: Optional[Path] = None):
        self.embedding_fn = embedding_fn
        self.db_path = path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self.engine)

    def add_entry(
        self,
        *,
        type: str,
        title: str,
        body: str,
        status: str = "open",
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        links: Optional[List[dict]] = None,
        revises_id: Optional[str] = None,
    ) -> Entry:
        entry_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        created = datetime.utcnow().replace(tzinfo=timezone.utc)
        row = EntryRow(
            id=entry_id,
            created_at=created,
            type=type,
            title=title,
            body_md=body,
            status=status,
            confidence=confidence,
            tags=",".join(tags) if tags else None,
            links=links,
            revises_id=revises_id,
        )
        vector = array("f", self.embedding_fn(body)).tobytes()
        with Session(self.engine) as sess:
            sess.add(row)
            sess.add(EmbeddingRow(entry_id=entry_id, vector=vector))
            sess.commit()
        entry = Entry(
            id=entry_id,
            type=type,
            title=title,
            body=body,
            created_at=created,
            status=status,
            confidence=confidence,
            tags=tags,
            links=links,
            revises_id=revises_id,
        )
        self._dump_markdown(entry)
        return entry

    def update_status(self, entry_id: str, status: str, confidence: Optional[float] = None) -> Entry:
        with Session(self.engine) as sess:
            row: EntryRow | None = sess.get(EntryRow, entry_id)
            if not row:
                raise KeyError(entry_id)
            entry = Entry(
                id=row.id,
                type=row.type,
                title=row.title,
                body=row.body_md,
                created_at=row.created_at,
                status=status,
                confidence=confidence if confidence is not None else row.confidence,
                tags=row.tags.split(",") if row.tags else None,
                links=row.links,
                revises_id=row.revises_id,
            )
        return self.add_entry(
            type=entry.type,
            title=entry.title,
            body=entry.body,
            status=status,
            confidence=confidence,
            tags=entry.tags,
            links=entry.links,
            revises_id=entry_id,
        )

    def list_entries(
        self,
        type: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Entry]:
        stmt = select(EntryRow)
        if type:
            stmt = stmt.where(EntryRow.type == type)
        if status:
            stmt = stmt.where(EntryRow.status == status)
        rows: Iterable[EntryRow]
        with Session(self.engine) as sess:
            rows = sess.scalars(stmt).all()
        entries = []
        for r in rows:
            if tag and (tag not in (r.tags or "")):
                continue
            entries.append(
                Entry(
                    id=r.id,
                    type=r.type,
                    title=r.title,
                    body=r.body_md,
                    created_at=r.created_at,
                    status=r.status,
                    confidence=r.confidence,
                    tags=r.tags.split(",") if r.tags else None,
                    links=r.links,
                    revises_id=r.revises_id,
                )
            )
        return entries

    def search(self, text: str, k: int = 5) -> List[SearchHit]:
        q_vec = np.array(self.embedding_fn(text), dtype=np.float32)
        with Session(self.engine) as sess:
            data = sess.execute(select(EmbeddingRow)).all()
        if not data:
            return []
        ids = []
        mat = []
        for row in data:
            ids.append(row.EmbeddingRow.entry_id if hasattr(row, 'EmbeddingRow') else row.entry_id)
            mat.append(np.frombuffer(row.EmbeddingRow.vector if hasattr(row, 'EmbeddingRow') else row.vector, dtype=np.float32))
        mat = np.vstack(mat)
        sims = mat @ q_vec / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q_vec) + 1e-9)
        top_idx = sims.argsort()[-k:][::-1]
        hits = []
        with Session(self.engine) as sess:
            for idx in top_idx:
                entry_row = sess.get(EntryRow, ids[idx])
                entry = Entry(
                    id=entry_row.id,
                    type=entry_row.type,
                    title=entry_row.title,
                    body=entry_row.body_md,
                    created_at=entry_row.created_at,
                    status=entry_row.status,
                    confidence=entry_row.confidence,
                    tags=entry_row.tags.split(",") if entry_row.tags else None,
                    links=entry_row.links,
                    revises_id=entry_row.revises_id,
                )
                hits.append(SearchHit(entry=entry, score=float(sims[idx])))
        return hits

    def _dump_markdown(self, entry: Entry) -> None:
        ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": entry.id,
            "type": entry.type,
            "title": entry.title,
            "authors": ["ai-assistant"],
            "date": entry.created_at.isoformat(),
            "status": entry.status,
            "confidence": entry.confidence,
            "tags": entry.tags,
            "links": entry.links,
            "revises_id": entry.revises_id,
        }
        md = "---\n" + json.dumps(meta, indent=2) + "\n---\n" + entry.body
        (ENTRIES_DIR / f"{entry.id}.md").write_text(md)
