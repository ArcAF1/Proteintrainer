"""Naïve entity + relation extractor used by the full data pipeline.

This keeps dependencies minimal: tries SciSpaCy; if unavailable falls back to
upper-case heuristics (very noisy but enough for a demo).
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Callable
import re


try:
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_sci_sm")  # will raise if model missing
except Exception:  # pragma: no cover
    _NLP = None


Ent = Dict[str, str]
Rel = Dict[str, str]


def _heuristic_extract(text: str) -> Tuple[List[Ent], List[Rel]]:
    sentences = re.split(r"[\.!?]\s+", text)
    ents: List[Ent] = []
    rels: List[Rel] = []
    for sent in sentences:
        caps = re.findall(r"\b([A-Z][a-z]{3,})\b", sent)[:2]
        if len(caps) == 2:
            e1, e2 = caps
            ents += [{"name": e1.lower(), "type": "Concept"}, {"name": e2.lower(), "type": "Concept"}]
            rels.append({
                "source": e1.lower(),
                "target": e2.lower(),
                "rel": "RELATED_TO",
                "type1": "Concept",
                "type2": "Concept",
            })
    return ents, rels


def extract(text: str) -> Tuple[List[Ent], List[Rel]]:
    if _NLP is None:
        return _heuristic_extract(text)
    doc = _NLP(text[:5000])
    ents: List[Ent] = [
        {"name": e.text.lower(), "type": e.label_ or "Concept"}
        for e in doc.ents
        if len(e.text) > 2
    ]
    rels: List[Rel] = []
    for sent in doc.sents:
        e_sent = [e for e in ents if e["name"] in sent.text.lower()]
        if len(e_sent) >= 2:
            for i in range(len(e_sent) - 1):
                rels.append({
                    "source": e_sent[i]["name"],
                    "type1": e_sent[i]["type"],
                    "rel": "MENTIONS",
                    "target": e_sent[i + 1]["name"],
                    "type2": e_sent[i + 1]["type"],
                })
    return ents, rels


def extract_triplets_bulk(docs: List[str], progress_cb: Callable[[float, str], None] | None = None):
    articles = []
    relations = []
    total = len(docs)
    for idx, doc in enumerate(docs):
        ents, rels = extract(doc)
        art_id = f"D{idx}"
        if ents:
            articles.append({"id": art_id, "title": f"Doc {idx}", "entities": ents})
        for r in rels:
            r["article_id"] = art_id
        relations.extend(rels)
        if progress_cb and idx % 100 == 0:
            progress_cb(idx / total, f"Entity extraction {idx}/{total}")
    return articles, relations 