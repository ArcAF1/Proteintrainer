"""Configuration for paths and model settings.

Usage example:
    from config import settings
    print(settings.data_dir)
"""
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Check if .env file exists, if not warn the user
if not Path('.env').exists() and not os.getenv('NEO4J_PASSWORD'):
    print("⚠️  Warning: No .env file found. Using default Neo4j password.")
    print("   Please copy env.template to .env and set a secure password.")


class Settings(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow field names like model_*
        arbitrary_types_allowed=True  # Allow Path types
    )
    
    data_dir: Path = Path("data")
    index_dir: Path = Path("indexes")
    model_dir: Path = Path("models")
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_model: str = "mistral-7b-instruct.Q4_0.gguf"
    top_k: int = 5
    chunk_size: int = 400
    
    # Neo4j settings - now properly secured
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "BioMed@2024!Research")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Connection retry settings
    neo4j_max_retries: int = 5
    neo4j_retry_delay: int = 5
    neo4j_connection_timeout: int = 60


settings = Settings()

