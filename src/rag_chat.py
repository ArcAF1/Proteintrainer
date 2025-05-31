"""RAG pipeline with robust error handling and graceful fallbacks.

Usage example:
    from rag_chat import answer
    response = await answer("What is aspirin?")
"""
from __future__ import annotations

import asyncio
import pickle
import traceback
import warnings
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from .config import settings
from .embeddings import Embedder
from .llm import get_llm


class RAGChat:
    """RAG system with robust error handling and fallbacks."""
    
    def __init__(self) -> None:
        self.embedder: Optional[Embedder] = None
        self.index: Optional[faiss.Index] = None
        self.docs: List[str] = []
        self.llm = None
        self.initialization_error: Optional[str] = None
        
        # Initialize components with error handling
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize RAG components with proper error handling."""
        try:
            # Initialize embedder
            try:
                self.embedder = Embedder()
                print("✅ Embedder initialized successfully")
            except Exception as e:
                self.initialization_error = f"Embedder initialization failed: {str(e)}"
                print(f"❌ Embedder error: {str(e)}")
                return
            
            # Prefer the *big* index; fall back to a mini index so the system
            # can answer something on a fresh clone.
            index_path = settings.index_dir / "pmc.faiss"
            store_path = settings.index_dir / "pmc.pkl"

            if not index_path.exists() or not store_path.exists():
                # Try mini fallback
                mini_index = settings.index_dir / "mini.faiss"
                mini_store = settings.index_dir / "mini.pkl"
                if mini_index.exists() and mini_store.exists():
                    index_path, store_path = mini_index, mini_store
                    warnings.warn(
                        "Using fallback mini index (limited knowledge base). "
                        "Run data-pipeline for full capabilities.")
                else:
                    self.initialization_error = (
                        "No search index found. Run the data pipeline to build "
                        "FAISS indexes (or run indexer.build_mini_index for a "
                        "quick demo)."
                    )
                    print("❌ No index files located – RAG disabled.")
                    return
            
            # Load FAISS index
            try:
                self.index = faiss.read_index(str(index_path))
                print(f"✅ FAISS index loaded: {self.index.ntotal:,} vectors")
            except Exception as e:
                self.initialization_error = f"FAISS index loading failed: {str(e)}"
                print(f"❌ FAISS error: {str(e)}")
                return
            
            # Load document store
            try:
                with open(store_path, "rb") as fh:
                    self.docs = pickle.load(fh)
                print(f"✅ Document store loaded: {len(self.docs):,} documents")
            except Exception as e:
                self.initialization_error = f"Document store loading failed: {str(e)}"
                print(f"❌ Document store error: {str(e)}")
                return
            
            # Initialize LLM
            try:
                model_path = settings.model_dir / settings.llm_model
                if not model_path.exists():
                    self.initialization_error = (
                        f"LLM model not found at {model_path}. "
                        "Please run training to download the model."
                    )
                    print(f"❌ LLM model missing: {model_path}")
                    return
                
                self.llm = get_llm()
                print("✅ LLM loaded successfully")
                
            except Exception as e:
                self.initialization_error = f"LLM loading failed: {str(e)}"
                print(f"❌ LLM error: {str(e)}")
                return
            
            print("🎉 RAG system fully initialized!")
            
        except Exception as e:
            self.initialization_error = f"RAG initialization failed: {str(e)}"
            print(f"❌ RAG initialization error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
    
    def is_ready(self) -> bool:
        """Check if RAG system is ready to answer questions."""
        return (
            self.embedder is not None and 
            self.index is not None and 
            self.docs and 
            self.llm is not None and
            self.initialization_error is None
        )
    
    def get_status(self) -> str:
        """Get detailed status of RAG components."""
        if self.is_ready():
            return f"✅ RAG System Ready\n" \
                   f"• Documents: {len(self.docs):,}\n" \
                   f"• Index size: {self.index.ntotal:,} vectors\n" \
                   f"• Model: {settings.llm_model}"
        
        status = "❌ RAG System Not Ready\n"
        
        # Check each component
        if self.embedder is None:
            status += "• ❌ Embedder not loaded\n"
        else:
            status += "• ✅ Embedder ready\n"
        
        if self.index is None:
            status += "• ❌ FAISS index not loaded\n"
        else:
            status += f"• ✅ FAISS index ready ({self.index.ntotal:,} vectors)\n"
        
        if not self.docs:
            status += "• ❌ Document store empty\n"
        else:
            status += f"• ✅ Document store ready ({len(self.docs):,} docs)\n"
        
        if self.llm is None:
            status += "• ❌ LLM not loaded\n"
        else:
            status += "• ✅ LLM ready\n"
        
        if self.initialization_error:
            status += f"\n🔍 Error Details:\n{self.initialization_error}"
        
        return status
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for the query."""
        if not self.is_ready():
            raise RuntimeError(f"RAG system not ready: {self.initialization_error}")
        
        try:
            vector = self.embedder.encode(query).reshape(1, -1)
            scores, ids = self.index.search(vector, settings.top_k)
            
            # Filter out invalid indices
            valid_docs = []
            for i in ids[0]:
                if 0 <= i < len(self.docs):
                    valid_docs.append(self.docs[i])
            
            return valid_docs
            
        except Exception as e:
            raise RuntimeError(f"Document retrieval failed: {str(e)}")
    
    async def generate(self, prompt: str) -> str:
        """Generate response using the LLM."""
        if not self.is_ready():
            raise RuntimeError(f"RAG system not ready: {self.initialization_error}")
        
        try:
            loop = asyncio.get_running_loop()
            
            # Truncate prompt if too long
            max_prompt_length = 3000
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            response = await loop.run_in_executor(None, self._safe_llm_call, prompt)
            return response
            
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def _safe_llm_call(self, prompt: str) -> str:
        """Safely call the LLM with timeout and error handling."""
        try:
            # Different interfaces for different LLM backends
            if hasattr(self.llm, 'create_completion'):
                # llama-cpp-python interface
                result = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=600,
                    temperature=0.7,
                    stop=["Human:", "Assistant:", "\n\n"],
                    echo=False
                )
                if isinstance(result, dict) and 'choices' in result:
                    return result['choices'][0]['text'].strip()
                return str(result).strip()
            elif hasattr(self.llm, '__call__'):
                # Direct callable interface - use create_completion instead
                result = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=600,
                    temperature=0.7,
                    stop=["Human:", "Assistant:", "\n\n"],
                    echo=False
                )
                if isinstance(result, dict) and 'choices' in result:
                    return result['choices'][0]['text'].strip()
                return str(result).strip()
            else:
                # ctransformers interface
                return self.llm(prompt, max_new_tokens=600, temperature=0.7)
                
        except Exception as e:
            error_str = str(e)
            if "llama_decode" in error_str:
                return "I'm experiencing technical difficulties with text generation. The model configuration may need adjustment."
            else:
                return f"Error generating response: {error_str}"
    
    async def answer(self, question: str) -> str:
        """Answer a question using RAG."""
        if not self.is_ready():
            return (
                f"❌ **RAG System Not Ready**\n\n"
                f"{self.get_status()}\n\n"
                f"**To fix this:**\n"
                f"1. Run system training to download models and build indexes\n"
                f"2. Check that all required files are present\n"
                f"3. Make sure you have enough disk space and memory"
            )
        
        try:
            # Retrieve relevant documents
            docs = self.retrieve(question)
            
            if not docs:
                return (
                    f"🔍 I couldn't find relevant information about: **{question}**\n\n"
                    f"This might be because:\n"
                    f"• The topic isn't in my knowledge base yet\n"
                    f"• The search index needs more training data\n"
                    f"• Try rephrasing your question\n\n"
                    f"💡 **Suggestion:** Try asking me to 'search the internet' for latest information!"
                )
            
            # Build context with citations
            cited = []
            context_parts = []
            for idx, doc in enumerate(docs, start=1):
                # Truncate very long documents
                truncated_doc = doc[:500] + "..." if len(doc) > 500 else doc
                context_parts.append(f"[{idx}] {truncated_doc}")
                cited.append(f"[{idx}]")
            
            # Ensure prompt won't exceed ~450 tokens (approx 800 words context + question)
            max_ctx_parts = context_parts.copy()
            while True:
                draft_context = "\n---\n".join(max_ctx_parts)
                approx_tokens = len(draft_context.split()) + len(question.split())
                if approx_tokens <= 450 or len(max_ctx_parts) == 1:
                    context_parts = max_ctx_parts
                    break
                # Drop the last doc to shrink prompt
                max_ctx_parts = max_ctx_parts[:-1]
            
            context = "\n---\n".join(context_parts)
            
            # Create prompt
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer (include citations like {', '.join(cited)} where relevant):"
            )
            
            # Generate response
            raw_response = await self.generate(prompt)
            
            # Clean up response
            if raw_response.startswith("Answer:"):
                raw_response = raw_response[7:].strip()
            
            # Add sources
            sources = "\n".join(context_parts)
            final_response = f"{raw_response}\n\n**Sources:**\n{sources}"
            
            return final_response
            
        except Exception as e:
            error_msg = str(e)
            print(f"RAG answer error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            
            return (
                f"❌ **Error answering question:** {error_msg}\n\n"
                f"**Troubleshooting:**\n"
                f"• Try asking me to 'test the system' to check components\n"
                f"• Restart the application if models are corrupted\n"
                f"• Re-run training if indexes are corrupted\n"
                f"• Check available memory and disk space"
            )


# Global instance
_chat: Optional[RAGChat] = None
_use_api_enhancement: bool = False  # Flag to enable API enhancement


def get_chat() -> RAGChat:
    """Get or create the global RAG chat instance."""
    global _chat
    if _chat is None:
        if _use_api_enhancement:
            try:
                from .api_enhanced_rag import APIEnhancedRAG
                _chat = APIEnhancedRAG(use_apis=True)
                print("[RAG] Using API-enhanced RAG with live data access")
            except ImportError:
                print("[RAG] API enhancement not available, using standard RAG")
                _chat = RAGChat()
        else:
            _chat = RAGChat()
    return _chat


def enable_api_enhancement(enabled: bool = True):
    """Enable or disable API enhancement for RAG."""
    global _use_api_enhancement, _chat
    _use_api_enhancement = enabled
    # Reset chat instance to apply change
    if _chat is not None:
        _chat = None
        

async def answer(question: str) -> str:
    """Answer a question using RAG with proper error handling."""
    try:
        chat = get_chat()
        return await chat.answer(question)
    except Exception as e:
        print(f"RAG answer top-level error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return (
            f"❌ **System Error:** {str(e)}\n\n"
            f"**What you can try:**\n"
            f"• Ask me to 'test the system' to diagnose issues\n"
            f"• Run 'start training' if components aren't built yet\n"
            f"• Restart the application to reset the system\n"
            f"• Check the console for detailed error messages"
        )


def get_rag_status() -> str:
    """Get the current status of the RAG system."""
    try:
        chat = get_chat()
        return chat.get_status()
    except Exception as e:
        return f"❌ Error checking RAG status: {str(e)}"




