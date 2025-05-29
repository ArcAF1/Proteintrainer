import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
pytest.importorskip("numpy")

import types

faiss = types.ModuleType("faiss")
ctransformers = types.ModuleType("ctransformers")
sys.modules.setdefault("faiss", faiss)
sys.modules.setdefault("ctransformers", ctransformers)

from src import rag_chat


@pytest.mark.asyncio
async def test_answer_returns_string():
    with patch.object(rag_chat, "get_chat") as mock_chat:
        mock_instance = AsyncMock()
        mock_instance.answer.return_value = "ok"
        mock_chat.return_value = mock_instance
        result = await rag_chat.answer("test question")
        assert isinstance(result, str)

