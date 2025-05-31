from __future__ import annotations
"""Biomedical relation extraction using a fine-tuned BioBERT model.

The model should be a sequence-classification checkpoint with label set matching
REL_TYPES in neo4j_schema.  Default model: `michiyasunaga/BioLinkBERT-base` fine-tuned on DDI.
"""
from typing import List, Tuple, Dict
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .neo4j_schema import REL_TYPES

MODEL_NAME = "michiyasunaga/BioLinkBERT-base"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_model.eval()
LABELS = [rel.lower() for rel in REL_TYPES]

def _candidate_pairs(sentence: str, entities: List[str]) -> List[Tuple[str, str]]:
    uniq = list(dict.fromkeys(entities))
    pairs = []
    for i, e1 in enumerate(uniq):
        for e2 in uniq[i + 1 :]:
            pairs.append((e1, e2))
    return pairs


def extract_relations(sentence: str, entities: List[str], threshold: float = 0.8) -> List[Dict]:
    results: List[Dict] = []
    for e1, e2 in _candidate_pairs(sentence, entities):
        text = sentence.replace(e1, "[E1]" + e1 + "[/E1]").replace(e2, "[E2]" + e2 + "[/E2]")
        inputs = _tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        if conf.item() >= threshold:
            results.append({
                "source": e1.lower(),
                "target": e2.lower(),
                "rel": LABELS[idx],
                "confidence": round(conf.item(), 3),
            })
    return results 