from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path
import types
import asyncio
import importlib
import pytest

sys.modules.setdefault('faiss', types.ModuleType('faiss'))
try:
    importlib.import_module('numpy')
except ImportError:
    pytest.skip("numpy missing", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_answer_returns_string():
    from src import rag_chat
    with patch.object(rag_chat, "get_chat") as mock_chat:
        mock_instance = AsyncMock()
        mock_instance.answer.return_value = "ok"
        mock_chat.return_value = mock_instance
        result = asyncio.run(rag_chat.answer("test question"))
        assert isinstance(result, str)
