"""Configuration for paths and model settings.

Usage example:
    from config import settings
    print(settings.data_dir)
"""
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("data")
    index_dir: Path = Path("indexes")
    model_dir: Path = Path("models")
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "mistral-7b-instruct.Q4_0.gguf"
    top_k: int = 5
    chunk_size: int = 400



    class Config:
        arbitrary_types_allowed = True


settings = Settings()

