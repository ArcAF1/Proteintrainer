"""Database models for research memory."""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    String,
    Float,
    Text,
    JSON,
    CheckConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Entry(Base):
    """Represents a notebook entry."""

    __tablename__ = "entry"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    type = Column(
        String,
        CheckConstraint(
            "type IN ('note','hypothesis','plan','observation','result','conclusion','question')"
        ),
        nullable=False,
    )
    title = Column(String, nullable=False)
    body_md = Column(Text, nullable=False)
    status = Column(String, default="open")
    confidence = Column(Float)
    tags = Column(String)
    links = Column(JSON)
    revises_id = Column(String, ForeignKey("entry.id"))

    revises = relationship("Entry", remote_side=[id])
    embedding = relationship("Embedding", uselist=False, back_populates="entry")


class Embedding(Base):
    """Stores embeddings for entries."""

    __tablename__ = "embedding"

    entry_id = Column(String, ForeignKey("entry.id"), primary_key=True)
    vector = Column(Text, nullable=False)

    entry = relationship("Entry", back_populates="embedding")
