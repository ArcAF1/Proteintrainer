import sys
from pathlib import Path
import types
sys.path.append(str(Path(__file__).resolve().parents[1]))

# stub numpy before importing library
numpy_stub = types.ModuleType('numpy')
numpy_stub.array = lambda x, dtype=None: x
numpy_stub.frombuffer = lambda b, dtype='float32': list(b)
numpy_stub.stack = lambda arrs: arrs
numpy_stub.linalg = types.SimpleNamespace(norm=lambda x, axis=None, keepdims=False: 1)
sys.modules.setdefault('numpy', numpy_stub)
sys.modules.setdefault('pydantic', types.SimpleNamespace(BaseModel=object))

import pytest
try:
    import sqlalchemy  # type: ignore
except Exception:
    pytest.skip("sqlalchemy not available", allow_module_level=True)

from research_memory import ResearchMemory
from research_memory.db import MemoryConfig
import research_memory.db as db


class NumpyStub(types.SimpleNamespace):
    def array(self, data, dtype=None):
        return data

    def frombuffer(self, buf, dtype="float32"):
        return list(buf)

    def stack(self, arrays):
        return arrays

    def __getattr__(self, name):
        if name == "linalg":
            return types.SimpleNamespace(norm=lambda x, axis=None, keepdims=False: 1)
        raise AttributeError

db.np = NumpyStub()

class DummyEmbed:
    def __call__(self, text: str):
        return [0.0, 1.0]


def test_add_and_search(tmp_path):
    cfg = MemoryConfig(db_path=tmp_path/"test.db", entries_dir=tmp_path/"entries")
    mem = ResearchMemory(embedding_fn=DummyEmbed(), config=cfg)
    # Add entry
    entry = mem.add_entry(type="note", title="t", body="hello")
    assert entry.id
    # Search
    res = mem.search("hello", k=1)
    assert res
    assert res[0][0].id == entry.id
    # Update
    updated = mem.update_status(entry.id, "supported", confidence=0.8)
    assert updated.revises_id == entry.id
