import importlib
import sys
from pathlib import Path
import types
import pytest

sys.modules.setdefault("neo4j", types.ModuleType("neo4j"))
pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_graph_modules_importable():
    importlib.import_module("src.graph_builder")
    importlib.import_module("src.graph_query")
    importlib.import_module("src.neo4j_setup")
