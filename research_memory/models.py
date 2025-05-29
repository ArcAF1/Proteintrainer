

"""Pydantic models for public API."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class EntryBase(BaseModel):
    id: str
    created_at: str
    type: str
    title: str
    body_md: str
    status: str
    confidence: Optional[float] = None
    tags: Optional[str] = None
    links: Optional[list] = None
    revises_id: Optional[str] = None


class EntryOut(EntryBase):
    pass


class EntryCreate(BaseModel):
    type: str
    title: str
    body: str
    status: str = "open"
    confidence: Optional[float] = None
    tags: Optional[List[str]] = None
    links: Optional[List[dict]] = None
    revises_id: Optional[str] = None


class SearchHit(BaseModel):
    entry: EntryOut
    score: float


