from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import importlib

if importlib.util.find_spec("numpy") is None:
    import pytest
    pytest.skip("numpy not installed", allow_module_level=True)

from research_memory import ResearchMemory


def fake_embed(text: str):
    return [0.0, 0.0, 0.0]


def test_add_and_search():
    with TemporaryDirectory() as tmp:
        mem = ResearchMemory(fake_embed, path=Path(tmp) / 'db.sqlite')
        mem.add_entry(type='note', title='t', body='b')
        hits = mem.search('b')
        assert hits


def test_update_immutable():
    with TemporaryDirectory() as tmp:
        mem = ResearchMemory(fake_embed, path=Path(tmp) / 'db.sqlite')
        e1 = mem.add_entry(type='note', title='t', body='b')
        e2 = mem.update_status(e1.id, 'supported', 0.9)
        assert e1.id != e2.id
        assert mem.list_entries()[0].id == e1.id
