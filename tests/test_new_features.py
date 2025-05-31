import importlib
import types
import sys
from pathlib import Path

# Stub heavy libs before import
sys.modules.setdefault("langchain", types.ModuleType("langchain"))
sys.modules.setdefault("langchain_community", types.ModuleType("langchain_community"))
sys.modules.setdefault("langchain_community.graphs", types.ModuleType("g"))
sys.modules.setdefault("langchain.chains", types.ModuleType("lc_chains"))
sys.modules.setdefault("langchain_community.llms", types.ModuleType("lcllms"))


def test_graph_rag_importable():
    mod = importlib.import_module("src.graph_rag")
    assert hasattr(mod, "GraphRAG")


def test_memory_save(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.memory_manager import save_finding, recall
    save_finding("q", "a")
    assert recall("q") 