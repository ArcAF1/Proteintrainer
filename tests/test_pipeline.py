import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import sys
from pathlib import Path
import types
import importlib

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# provide fake faiss module if not installed
if "faiss" not in sys.modules:
    sys.modules["faiss"] = types.ModuleType("faiss")

if "numpy" not in sys.modules:
    fake_np = types.ModuleType("numpy")
    class ndarray(list):
        pass

    def array(val, dtype=None):
        return ndarray(val)

    fake_np.ndarray = ndarray
    fake_np.array = array
    sys.modules["numpy"] = fake_np

if "ctransformers" not in sys.modules:
    fake_ct = types.ModuleType("ctransformers")
    class AutoModelForCausalLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt: str, **kwargs):
            return {"choices": [{"text": "mock"}]}

    fake_ct.AutoModelForCausalLM = AutoModelForCausalLM
    sys.modules["ctransformers"] = fake_ct

if "pydantic" not in sys.modules:
    fake_pd = types.ModuleType("pydantic")
    class BaseModel:
        pass
    fake_pd.BaseModel = BaseModel
    sys.modules["pydantic"] = fake_pd

if "sentence_transformers" not in sys.modules:
    fake_st = types.ModuleType("sentence_transformers")
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, text, normalize_embeddings=True):
            if isinstance(text, list):
                return [[0.0]] * len(text)
            return [0.0]

    fake_st.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = fake_st

rag_chat = importlib.import_module("src.rag_chat")


@pytest.mark.asyncio
async def test_answer_returns_string():
    with patch.object(rag_chat, "get_chat") as mock_chat:
        mock_instance = AsyncMock()
        mock_instance.answer.return_value = "ok"
        mock_chat.return_value = mock_instance
        result = await rag_chat.answer("test question")
        assert isinstance(result, str)

