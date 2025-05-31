"""Sentence-Transformers embedding wrapper with robust error handling.

Usage example:
    from embeddings import Embedder
    embedder = Embedder()
    if embedder.is_ready():
        vec = embedder.encode("hello")
"""

import os
import traceback
from typing import Optional, Union, List, Tuple
from pathlib import Path
import logging

import numpy as np

from .config import settings

# Try to import sentence-transformers with error handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ sentence-transformers library loaded successfully")
except ImportError as e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"❌ sentence-transformers not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    """Lightweight embedding helper with robust error handling."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.model_name = model_name or settings.embed_model
        self.initialization_error = None
        self.dimension = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the embedding model with error handling."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.initialization_error = "sentence-transformers library not available"
            logger.error(self.initialization_error)
            return

        try:
            print(f"Loading embedding model: {self.model_name}")
            
            # FORCE CPU ONLY - NO GPU/MPS to prevent system overload
            device = "cpu"
            print(f"🛡️  FORCING CPU-only mode for embeddings to prevent system overload")
            
            # Check if running in an environment with limited internet
            cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
            
            # Create cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to load the model with device specification
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(cache_dir),
                device=device  # ALWAYS CPU
            )
            
            print(f"✅ Embedding model loaded successfully: {self.model_name} on {device}")
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Embedding model loading failed: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Provide helpful error message based on the type of error
            if "ConnectionError" in error_msg or "HTTP" in error_msg:
                self.initialization_error = (
                    f"Network error loading embedding model '{self.model_name}'. "
                    "Please check your internet connection and try again."
                )
            elif "No space left" in error_msg or "disk" in error_msg.lower():
                self.initialization_error = (
                    f"Insufficient disk space to download embedding model '{self.model_name}'. "
                    "Please free up some space and try again."
                )
            elif "permission" in error_msg.lower():
                self.initialization_error = (
                    f"Permission error loading embedding model '{self.model_name}'. "
                    "Please check file permissions on the cache directory."
                )
            else:
                self.initialization_error = (
                    f"Failed to load embedding model '{self.model_name}': {error_msg}"
                )
            self.model = None
            self.dimension = None
            logger.error(f"Embedding model import failed: {self.initialization_error}")
    
    def is_ready(self) -> bool:
        """Check if the embedder is ready to use."""
        return self.model is not None and self.initialization_error is None
    
    def get_status(self) -> str:
        """Get the status of the embedder."""
        if self.is_ready():
            return f"✅ Embedder ready: {self.model_name}"
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return "❌ sentence-transformers library not installed"
        
        return f"❌ Embedder failed: {self.initialization_error}"
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text with robust error handling and dimension validation."""
        if not self.is_ready():
            raise RuntimeError(f"Embedder not ready: {self.initialization_error}")
        
        # Ensure input is properly formatted
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("Input must be string or list of strings")
        
        # Filter out empty or invalid inputs
        valid_texts = [t for t in text if t and isinstance(t, str) and len(t.strip()) > 0]
        if not valid_texts:
            logger.warning("No valid texts to encode, returning zero vector")
            return np.zeros((1, self.dimension))
        
        try:
            vector = self.model.encode(valid_texts, convert_to_numpy=True)
            
            # Ensure correct shape and dimensions
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            # Validate dimensions match expected
            if vector.shape[1] != self.dimension:
                logger.error(f"Dimension mismatch: expected {self.dimension}, got {vector.shape[1]}")
                raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {vector.shape[1]}")
            
            # Normalize for cosine similarity if needed
            from sklearn.preprocessing import normalize
            vector = normalize(vector, norm='l2')
            
            return vector
        except Exception as e:
            logger.error(f"Encoding failed for {len(valid_texts)} texts: {e}")
            # Return zero vector with correct dimensions
            return np.zeros((len(valid_texts), self.dimension))
    
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        if not self.is_ready():
            raise RuntimeError(f"Embedder not ready: {self.initialization_error}")
        
        try:
            # Clean inputs
            cleaned_texts = []
            for text in texts:
                if not isinstance(text, str):
                    text = str(text)
                if not text.strip():
                    text = " "  # Placeholder for empty text
                # Truncate long texts
                if len(text) > 512:
                    text = text[:512]
                cleaned_texts.append(text)
            
            # Encode batch
            vectors = self.model.encode(
                cleaned_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32  # Reasonable batch size
            )
            
            return np.array(vectors, dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Batch encoding failed: {str(e)}")

class RobustRAGInitializer:
    """Robust RAG initialization with comprehensive error handling."""
    
    def __init__(self, embedding_model_name: str = None):
        self.embedding_model_name = embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.embedder = None
        self.dimension = None
        self._initialize_embedder()
        
    def _initialize_embedder(self):
        """Initialize embedder with fallback models."""
        fallback_models = [
            self.embedding_model_name,
            "all-MiniLM-L6-v2", 
            "paraphrase-MiniLM-L6-v2"
        ]
        
        for model_name in fallback_models:
            try:
                logger.info(f"Trying to load embedding model: {model_name}")
                self.embedder = Embedder(model_name)
                if self.embedder.is_ready():
                    self.dimension = self.embedder.dimension
                    logger.info(f"✅ Successfully loaded {model_name} (dim: {self.dimension})")
                    return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
                
        raise RuntimeError("Failed to load any embedding model")
    
    def validate_and_encode(self, documents: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Encode documents with validation and error recovery."""
        if not self.embedder.is_ready():
            raise RuntimeError("Embedder not initialized")
            
        valid_docs = []
        valid_embeddings = []
        
        for i, doc in enumerate(documents):
            try:
                # Basic validation
                if not doc or not isinstance(doc, str) or len(doc.strip()) < 10:
                    logger.debug(f"Skipping invalid document {i}: too short or empty")
                    continue
                    
                # Encode with dimension validation
                embedding = self.embedder.encode(doc)
                
                if embedding.shape[1] != self.dimension:
                    logger.warning(f"Skipping document {i}: dimension mismatch")
                    continue
                    
                valid_embeddings.append(embedding[0])  # Remove batch dimension
                valid_docs.append(doc)
                
            except Exception as e:
                logger.warning(f"Error processing document {i}: {e}")
                continue
        
        if not valid_embeddings:
            raise ValueError("No valid documents could be encoded")
            
        embeddings_array = np.array(valid_embeddings).astype('float32')
        
        # Normalize for cosine similarity
        import faiss
        faiss.normalize_L2(embeddings_array)
        
        return embeddings_array, valid_docs
    
    def create_faiss_index(self, embeddings: np.ndarray = None) -> 'faiss.Index':
        """Create FAISS index with validated dimensions."""
        import faiss
        
        dimension = embeddings.shape[1] if embeddings is not None else self.dimension
        
        # Use IndexFlatIP for cosine similarity (normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        
        if embeddings is not None:
            index.add(embeddings)
            logger.info(f"✅ Created FAISS index with {embeddings.shape[0]} vectors, dim={dimension}")
            
        return index

# Global embedder instance - lazy loaded (after class definition)
_embedder: Optional[Embedder] = None

