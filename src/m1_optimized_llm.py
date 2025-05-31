"""
M1-Optimized LLM Loader with Metal Acceleration
"""
from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class M1OptimizedLLM:
    """Optimized LLM loader for Apple Silicon Macs."""
    
    def __init__(self, config_path: str = "m1_optimized_config.json"):
        self.config = self._load_config(config_path)
        self.llm = None
        self.backend = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load optimized configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default M1 optimized configuration."""
        return {
            "llm_config": {
                "n_gpu_layers": 24,
                "n_ctx": 4096,
                "n_batch": 512,
                "use_metal": True,
                "n_threads": 6,
                "use_mmap": True,
                "f16_kv": True
            }
        }
        
    def _detect_best_backend(self) -> str:
        """Detect the best backend for current system."""
        system = platform.system()
        processor = platform.processor()
        
        if system == "Darwin" and "arm" in processor.lower():
            # Check for available backends
            try:
                import mlx
                logger.info("MLX available - using Apple's optimized framework")
                return "mlx"
            except ImportError:
                pass
                
            try:
                from llama_cpp import Llama
                # Check if Metal support is compiled in
                test_llm = Llama(
                    model_path="models/test.gguf",
                    n_gpu_layers=1,
                    n_ctx=512,
                    verbose=False
                )
                if hasattr(test_llm, 'metal_total_size'):
                    logger.info("llama.cpp with Metal support detected")
                    return "llama_cpp_metal"
            except:
                pass
                
        return "llama_cpp"
        
    def load_model(self, model_path: Optional[str] = None) -> Any:
        """Load model with M1 optimizations."""
        if model_path is None:
            model_path = self._select_best_model()
            
        self.backend = self._detect_best_backend()
        
        if self.backend == "mlx":
            return self._load_mlx_model(model_path)
        elif self.backend == "llama_cpp_metal":
            return self._load_llama_cpp_metal(model_path)
        else:
            return self._load_llama_cpp(model_path)
            
    def _select_best_model(self) -> str:
        """Select the best model for M1 performance."""
        models_dir = Path("models")
        
        # Priority order for M1 (balancing quality and speed)
        preferred_models = [
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Best balance
            "mistral-7b-instruct-v0.2.Q5_K_M.gguf",  # Higher quality
            "mistral-7b-instruct.Q4_0.gguf",         # Fallback
            "phi-2.Q4_K_M.gguf",                      # Smaller, faster
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"   # Tiny but fast
        ]
        
        for model_name in preferred_models:
            model_path = models_dir / model_name
            if model_path.exists():
                logger.info(f"Selected model: {model_name}")
                return str(model_path)
                
        # Return first available model
        gguf_files = list(models_dir.glob("*.gguf"))
        if gguf_files:
            return str(gguf_files[0])
            
        raise FileNotFoundError("No models found. Run training first.")
        
    def _load_mlx_model(self, model_path: str) -> Any:
        """Load model using MLX (Apple's framework)."""
        try:
            from mlx_lm import load, generate
            
            model, tokenizer = load(model_path)
            
            # Wrap in a compatible interface
            class MLXWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    
                def create_completion(self, prompt: str, **kwargs):
                    max_tokens = kwargs.get('max_tokens', 512)
                    temperature = kwargs.get('temperature', 0.7)
                    
                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temp=temperature,
                        verbose=False
                    )
                    
                    return {
                        'choices': [{'text': response}]
                    }
                    
            logger.info("Model loaded with MLX backend")
            return MLXWrapper(model, tokenizer)
            
        except Exception as e:
            logger.error(f"MLX loading failed: {e}")
            return self._load_llama_cpp_metal(model_path)
            
    def _load_llama_cpp_metal(self, model_path: str) -> Any:
        """Load model with llama.cpp + Metal acceleration."""
        from llama_cpp import Llama
        
        config = self.config['llm_config']
        
        # Enable all Metal optimizations
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=config.get('n_gpu_layers', 24),
            n_ctx=config.get('n_ctx', 4096),
            n_batch=config.get('n_batch', 512),
            n_threads=config.get('n_threads', 6),
            f16_kv=config.get('f16_kv', True),
            use_mlock=config.get('use_mlock', True),
            use_mmap=config.get('use_mmap', True),
            offload_kqv=True,  # Offload KV cache to GPU
            logits_all=False,
            vocab_only=False,
            verbose=False,
            # Metal-specific optimizations
            mul_mat_q=True,  # Use quantized matrix multiplication
            tensor_split=None,  # Let Metal handle tensor splitting
        )
        
        logger.info(f"Model loaded with Metal acceleration: {config['n_gpu_layers']} layers on GPU")
        return llm
        
    def _load_llama_cpp(self, model_path: str) -> Any:
        """Fallback CPU loading (still optimized for M1)."""
        from llama_cpp import Llama
        
        config = self.config['llm_config']
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=0,  # CPU only
            n_ctx=config.get('n_ctx', 2048),
            n_batch=config.get('n_batch', 256),
            n_threads=8,  # Use all efficiency cores
            f16_kv=False,
            use_mlock=True,
            use_mmap=True,
            verbose=False
        )
        
        logger.info("Model loaded in CPU mode (Metal not available)")
        return llm
        
    def create_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion with automatic optimization."""
        if self.llm is None:
            self.llm = self.load_model()
            
        # Apply inference optimizations
        optimized_kwargs = self._optimize_generation_params(kwargs)
        
        return self.llm.create_completion(prompt, **optimized_kwargs)
        
    def _optimize_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize generation parameters for speed."""
        optimizations = self.config.get('inference_optimizations', {})
        
        # Set optimal defaults
        kwargs.setdefault('max_tokens', 512)
        kwargs.setdefault('temperature', 0.7)
        kwargs.setdefault('top_p', 0.9)
        kwargs.setdefault('repeat_penalty', 1.1)
        
        # Enable caching
        if optimizations.get('prompt_cache', True):
            kwargs['cache_prompt'] = True
            
        # Use continuous batching
        if optimizations.get('continuous_batching', True):
            kwargs['stream'] = False  # Better for batching
            
        return kwargs


def get_optimized_llm() -> M1OptimizedLLM:
    """Get the optimized LLM instance."""
    return M1OptimizedLLM()


def benchmark_inference(prompt: str = "What is creatine?") -> Dict[str, float]:
    """Benchmark inference speed."""
    import time
    
    llm = get_optimized_llm()
    
    # Warmup
    llm.create_completion(prompt, max_tokens=10)
    
    # Benchmark
    times = []
    for _ in range(5):
        start = time.time()
        response = llm.create_completion(prompt, max_tokens=200)
        elapsed = time.time() - start
        times.append(elapsed)
        
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'tokens_per_second': 200 / (sum(times) / len(times))
    } 