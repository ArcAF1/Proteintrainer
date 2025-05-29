"""Database models and session helpers."""
from __future__ import annotations

from pathlib import Path
from sqlalchemy import (
    Column,
    String,
    Float,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    LargeBinary,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import datetime

Base = declarative_base()


class Entry(Base):
    __tablename__ = "entry"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.utcnow())
    type = Column(String)
    title = Column(String)
    body_md = Column(Text)
    status = Column(String)
    confidence = Column(Float, nullable=True)
    tags = Column(String, nullable=True)
    links = Column(JSON, nullable=True)
    revises_id = Column(String, ForeignKey("entry.id"), nullable=True)
    previous = relationship("Entry", remote_side=[id])


class Embedding(Base):
    __tablename__ = "embedding"
    entry_id = Column(String, ForeignKey("entry.id"), primary_key=True)
    vector = Column(LargeBinary)
    entry = relationship("Entry", backref="embedding")


def get_engine(path: Path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", future=True)


def get_session(engine):
    return sessionmaker(bind=engine, future=True)()
