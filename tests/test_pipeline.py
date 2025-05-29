import sys
from pathlib import Path
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import rag_chat


@pytest.mark.asyncio
async def test_answer_returns_string():
    with patch.object(rag_chat, "get_chat") as mock_chat:
        mock_instance = AsyncMock()
        mock_instance.answer.return_value = "ok"
        mock_chat.return_value = mock_instance
        result = await rag_chat.answer("test question")
        assert isinstance(result, str)

