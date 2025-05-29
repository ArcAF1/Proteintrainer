import sys
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).resolve().parent.parent))
from research_memory import ResearchMemory


def dummy_embed(text: str):
    return [1.0, 0.0]


def test_add_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        mem = ResearchMemory(embedding_fn=dummy_embed, db_path=f"{tmp}/db.sqlite")
        entry = mem.add_entry(type="note", title="t", body="hello world")
        hits = mem.search("hello", k=1)
        assert hits
        assert hits[0].entry.id == entry.id


def test_update_immutability():
    with tempfile.TemporaryDirectory() as tmp:
        mem = ResearchMemory(embedding_fn=dummy_embed, db_path=f"{tmp}/db.sqlite")
        e1 = mem.add_entry(type="note", title="t", body="body")
        e2 = mem.update_status(e1.id, "closed", confidence=0.5)
        assert e1.id != e2.id
        entries = mem.list_entries()
        assert len(entries) == 2
