from __future__ import annotations
"""Simple biomedical relation extraction helpers.

For demo purposes this module tries to load a SciSpaCy model (en_core_sci_md).  
If that fails (e.g. model not installed), it falls back to a *very* naive
regex-based extractor that just looks for pairs of capitalised words in the
same sentence.

Returned schema matches what `graph_builder.build_graph()` expects.
"""

from typing import List, Dict
import re


def _extract_with_regex(text: str) -> tuple[List[Dict], List[Dict]]:
    """Extremely naive fallback extraction."""
    sentences = re.split(r"[.!?]\s+", text)
    entities: List[Dict] = []
    relations: List[Dict] = []
    for sent in sentences:
        caps = re.findall(r"\b([A-Z][a-z]{3,})\b", sent)[:2]
        if len(caps) == 2:
            e1, e2 = caps
            entities.extend([
                {"name": e1.lower(), "type": "Concept"},
                {"name": e2.lower(), "type": "Concept"},
            ])
            relations.append({
                "source": e1.lower(),
                "target": e2.lower(),
                "rel": "RELATED_TO",
                "type1": "Concept",
                "type2": "Concept",
            })
    return entities, relations


def extract(text: str) -> tuple[List[Dict], List[Dict]]:  # noqa: D401
    """Return (entities, relations) extracted from *text*."""
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_sci_md")  # type: ignore
        except Exception:  # pragma: no cover
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)
        ents: List[Dict] = []
        for ent in doc.ents:
            ents.append({"name": ent.text.lower(), "type": ent.label_ or "Concept"})

        rels: List[Dict] = []
        # placeholder: pair every two entities in same sentence
        for sent in doc.sents:
            e_in_sent = [e for e in ents if e["name"] in sent.text.lower()]
            if len(e_in_sent) >= 2:
                for i in range(len(e_in_sent) - 1):
                    rels.append({
                        "source": e_in_sent[i]["name"],
                        "target": e_in_sent[i + 1]["name"],
                        "rel": "RELATED_TO",
                        "type1": e_in_sent[i]["type"],
                        "type2": e_in_sent[i + 1]["type"],
                    })
        return ents, rels
    except Exception:  # pragma: no cover
        return _extract_with_regex(text) 