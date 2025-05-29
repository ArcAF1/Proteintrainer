from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict


@dataclass
class Entry:
    """Single notebook entry."""

    id: str
    type: str
    title: str
    body: str
    created_at: datetime
    status: str
    confidence: Optional[float] = None
    tags: Optional[List[str]] = None
    links: Optional[List[Dict]] = None
    revises_id: Optional[str] = None


@dataclass
class SearchHit:
    entry: Entry
    score: float
