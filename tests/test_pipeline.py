import asyncio
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import types
numpy_stub = types.ModuleType('numpy')
numpy_stub.array = lambda x, dtype=None: x
numpy_stub.frombuffer = lambda b, dtype='float32': b
numpy_stub.stack = lambda arrs: arrs
numpy_stub.linalg = types.SimpleNamespace(norm=lambda x, axis=None, keepdims=False: 1)
sys.modules.setdefault('numpy', numpy_stub)
sys.modules.setdefault('pydantic', types.SimpleNamespace(BaseModel=object))
sys.modules.setdefault("faiss", types.SimpleNamespace(IndexFlatIP=object, read_index=lambda *a, **k: object()))
sys.modules.setdefault("ctransformers", types.SimpleNamespace(AutoModelForCausalLM=object))
try:
    import sentence_transformers  # type: ignore
except Exception:
    import pytest
    pytest.skip("sentence-transformers missing", allow_module_level=True)

import pytest

from src import rag_chat


@pytest.mark.asyncio
async def test_answer_returns_string():
    with patch.object(rag_chat, "get_chat") as mock_chat:
        mock_instance = AsyncMock()
        mock_instance.answer.return_value = "ok"
        mock_chat.return_value = mock_instance
        result = await rag_chat.answer("test question")
        assert isinstance(result, str)

