import asyncio
from unittest.mock import AsyncMock, patch
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("faiss")
from src import rag_chat


@pytest.mark.asyncio
async def test_answer_returns_citation():
    with patch.object(rag_chat, "get_chat") as mock_chat:
        mock_instance = AsyncMock()
        mock_instance.answer.return_value = "(Dok 1) example"
        mock_chat.return_value = mock_instance
        result = await rag_chat.answer("test question")
        assert "(Dok" in result

