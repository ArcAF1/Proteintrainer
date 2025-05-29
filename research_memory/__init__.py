


"""Electronic Lab Notebook interface for the RAG assistant.

Usage example:
    from research_memory import ResearchMemory
    memory = ResearchMemory(embedding_fn=lambda t: [0.0])
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List

from .db import Base, Entry, Embedding, get_engine, get_session
from .models import EntryOut, SearchHit

import datetime
import uuid
import json
import numpy as np
from array import array


class ResearchMemory:
    """Main interface for storing and searching research notes."""

    def __init__(
        self,
        embedding_fn: Callable[[str], List[float]],
        db_path: str | Path | None = None,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.db_path = Path(db_path or Path.home() / "research_memory" / "research.db").expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries_dir = self.db_path.parent / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.engine = get_engine(self.db_path)
        Base.metadata.create_all(self.engine)

    def _session(self):
        return get_session(self.engine)

    def _new_id(self) -> str:
        return datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def add_entry(
        self,
        *,
        type: str,
        title: str,
        body: str,
        status: str = "open",
        confidence: float | None = None,
        tags: List[str] | None = None,
        links: List[dict] | None = None,
        revises_id: str | None = None,
    ) -> EntryOut:
        entry_id = self._new_id()
        created_at = datetime.datetime.utcnow()
        tags_str = ",".join(tags) if tags else None
        links_json = json.dumps(links) if links else None
        front_matter = {
            "id": entry_id,
            "type": type,
            "title": title,
            "authors": ["ai-assistant"],
            "date": created_at.isoformat() + "Z",
            "status": status,
            "confidence": confidence,
            "tags": tags or [],
            "links": links or [],
        }
        file_path = self.entries_dir / f"{entry_id}.md"
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("---\n")
            fh.write(json.dumps(front_matter, indent=2))
            fh.write("\n---\n")
            fh.write(body)
        vector = array("f", self.embedding_fn(body)).tobytes()
        with self._session() as ses:
            row = Entry(
                id=entry_id,
                created_at=created_at,
                type=type,
                title=title,
                body_md=body,
                status=status,
                confidence=confidence,
                tags=tags_str,
                links=links_json,
                revises_id=revises_id,
            )
            ses.add(row)
            ses.flush()
            ses.add(Embedding(entry_id=entry_id, vector=vector))
            ses.commit()
            ses.refresh(row)
            return EntryOut.model_validate(row)

    def update_status(self, entry_id: str, status: str, confidence: float | None = None) -> EntryOut:
        with self._session() as ses:
            old = ses.get(Entry, entry_id)
            if not old:
                raise KeyError(entry_id)
            new_row = Entry(
                id=self._new_id(),
                created_at=datetime.datetime.utcnow(),
                type=old.type,
                title=old.title,
                body_md=old.body_md,
                status=status,
                confidence=confidence if confidence is not None else old.confidence,
                tags=old.tags,
                links=old.links,
                revises_id=old.id,
            )
            ses.add(new_row)
            ses.commit()
            ses.refresh(new_row)
            return EntryOut.model_validate(new_row)

    def _search_vectors(self, vector: np.ndarray, k: int) -> List[tuple[Entry, float]]:
        with self._session() as ses:
            all_embeds = ses.query(Embedding).all()
            if not all_embeds:
                return []
            mat = np.stack([np.frombuffer(e.vector, dtype="float32") for e in all_embeds])
            vec_norm = vector / np.linalg.norm(vector)
            mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
            scores = mat_norm @ vec_norm
            ids = np.argsort(-scores)[:k]
            hits = []
            for i in ids:
                entry = ses.get(Entry, all_embeds[i].entry_id)
                hits.append((entry, float(scores[i])))
            return hits

    def search(self, text: str, k: int = 5) -> List[SearchHit]:
        vector = np.array(self.embedding_fn(text), dtype="float32")
        hits = self._search_vectors(vector, k)
        return [SearchHit(entry=EntryOut.model_validate(e), score=s) for e, s in hits]

    def list_entries(
        self,
        type: str | None = None,
        status: str | None = None,
        tag: str | None = None,
    ) -> List[EntryOut]:
        with self._session() as ses:
            query = ses.query(Entry)
            if type:
                query = query.filter(Entry.type == type)
            if status:
                query = query.filter(Entry.status == status)
            if tag:
                query = query.filter(Entry.tags.like(f"%{tag}%"))
            rows = query.order_by(Entry.created_at).all()
            return [EntryOut.model_validate(r) for r in rows]


__all__ = ["ResearchMemory", "EntryOut", "SearchHit"]


