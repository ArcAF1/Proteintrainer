from __future__ import annotations
"""LLM loader that prefers llama-cpp-python (Metal) then ctransformers."""
from pathlib import Path
from .config import settings
import json
import os
import platform

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover
    Llama = None  # type: ignore

try:
    from ctransformers import AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore

# Optional heavy backend – only used when LoRA adapters present
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM as HFModel
    from peft import PeftModel
    import torch
except Exception:  # pragma: no cover
    HFModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    PeftModel = None  # type: ignore
    torch = None  # type: ignore

# Import M1 optimized loader
try:
    from .m1_optimized_llm import get_optimized_llm
    M1_OPTIMIZED_AVAILABLE = True
except ImportError:
    M1_OPTIMIZED_AVAILABLE = False


def get_cpu_only_config():
    """Get CPU-only configuration for memory-constrained systems."""
    return {
        "n_gpu_layers": 0,       # CPU only - NO GPU
        "n_ctx": 256,            # Very small context
        "n_batch": 8,            # Tiny batch
        "f16_kv": False,         # Disabled
        "use_mlock": False,      # Disabled
        "n_threads": 1,          # Single thread only
        "verbose": False,
        "low_vram": True,
        "use_mmap": True,
        "numa": False,
    }

def load_macbook_config():
    """Load MacBook-specific configuration."""
    config_file = Path("macbook_config.json")
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                return config.get("llm_config", {})
        except Exception as e:
            print(f"Warning: Could not load MacBook config: {e}")
    return {}

def get_macbook_optimized_config():
    """Get ultra-conservative configuration optimized for MacBook Pro 13-inch M1 with other apps running."""
    base_config = {
        "n_gpu_layers": 0,       # CPU ONLY - absolutely no GPU
        "n_ctx": 1024,           # Enough for ~800 tokens
        "n_batch": 2,            # Minimal batch size
        "f16_kv": False,         # Disabled for stability
        "use_mlock": False,      # Don't lock memory
        "n_threads": 1,          # Single thread to leave CPU for other apps
        "verbose": False,
        "low_vram": True,
        "use_mmap": True,        # Memory-efficient file access
        "numa": False,
        "max_tokens": 384,       # Half-page answers
        "temperature": 0.3,      # Lower creativity for efficiency
        "top_p": 0.8,           # Focused sampling
    }
    
    # Override with MacBook-specific settings if available
    macbook_config = load_macbook_config()
    base_config.update(macbook_config)
    
    return base_config

def get_ultra_conservative_config():
    """Ultra-conservative configuration to prevent system overload."""
    return get_macbook_optimized_config()  # Use MacBook-optimized config

def get_optimal_m1_config():
    """Get CPU-only configuration - NO GPU to prevent overload."""
    return get_macbook_optimized_config()  # Use MacBook-optimized config

def get_medicine_llm_config():
    """Get CPU-only configuration for Medicine LLM - NO GPU."""
    return get_macbook_optimized_config()  # Use MacBook-optimized config

def detect_model_type(model_path: str) -> str:
    """Detect the type of model based on filename."""
    model_name = Path(model_path).name.lower()
    
    if "medicine" in model_name:
        return "medicine-llm"
    elif "pmc-llama" in model_name:
        return "pmc-llama"
    elif "mistral" in model_name:
        return "mistral"
    else:
        return "unknown"

def get_optimal_config_for_model(model_path: str):
    """Get optimal configuration based on the model type."""
    model_type = detect_model_type(model_path)
    
    if model_type == "medicine-llm":
        print("[llm] Using Medicine LLM optimized configuration")
        return get_medicine_llm_config()
    elif model_type in ["pmc-llama", "mistral"]:
        print(f"[llm] Using standard configuration for {model_type}")
        return get_optimal_m1_config()
    else:
        print("[llm] Using default configuration for unknown model")
        return get_optimal_m1_config()

def is_apple_silicon():
    """Check if running on Apple Silicon (M1/M2)."""
    if platform.system() != "Darwin":
        return False
    
    processor = platform.processor()
    return "arm" in processor.lower() or "apple" in processor.lower()

def get_llm():
    # Check if we're on Apple Silicon and should use optimized loader
    if is_apple_silicon() and M1_OPTIMIZED_AVAILABLE:
        try:
            print("[llm] Detected Apple Silicon - using M1-optimized loader")
            optimized_llm = get_optimized_llm()
            return optimized_llm.load_model()
        except Exception as e:
            print(f"[llm] M1-optimized loader failed: {e}, falling back to standard loader")
    
    models_dir = Path(settings.model_dir)
    
    # Find all available GGUF models
    available_models = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
    
    if not available_models:
        raise RuntimeError(f"No GGUF models found in {models_dir}. Please run installation or download models.")
    
    # Check if we're in MacBook-optimized mode
    macbook_mode = os.getenv('MACBOOK_OPTIMIZED') == '1'
    
    if macbook_mode:
        # MacBook-specific model priority: prefer smaller, efficient models
        print("[llm] MacBook mode detected - prioritizing smaller models for stability")
        
        # Filter models by size for MacBook (prefer models under 5GB)
        small_models = []
        large_models = []
        
        for model_path in available_models:
            size_gb = model_path.stat().st_size / (1024**3)
            if size_gb <= 5.0:  # 5GB or smaller
                small_models.append((model_path, size_gb))
            else:
                large_models.append((model_path, size_gb))
        
        # Sort small models by size (largest small model first for best quality)
        small_models.sort(key=lambda x: x[1], reverse=True)
        
        # Prefer small models, but fall back to large if needed
        if small_models:
            selected_model = small_models[0][0]
            size_gb = small_models[0][1]
            print(f"[llm] Selected MacBook-optimized model: {selected_model.name} ({size_gb:.1f}GB)")
        else:
            # If no small models, use the smallest large model
            large_models.sort(key=lambda x: x[1])  # smallest first
            selected_model = large_models[0][0]
            size_gb = large_models[0][1]
            print(f"[llm] WARNING: Using large model {selected_model.name} ({size_gb:.1f}GB) - may impact performance")
    else:
        # Standard priority order: Medicine LLM > PMC-LLaMA > Mistral > Others
        model_priority = ["medicine", "pmc-llama", "mistral"]
        
        selected_model = None
        
        # Find the highest priority model
        for priority_type in model_priority:
            for model_path in available_models:
                if priority_type in model_path.name.lower():
                    selected_model = model_path
                    break
            if selected_model:
                break
        
        # If no priority model found, use the largest model
        if not selected_model:
            selected_model = max(available_models, key=lambda x: x.stat().st_size)
    
    print(f"[llm] Selected model: {selected_model.name}")
    size_gb = selected_model.stat().st_size / (1024**3)
    print(f"[llm] Model size: {size_gb:.1f} GB")

    # Skip LoRA adapters in MacBook mode for simplicity and stability
    if not macbook_mode:
        # 1) Prefer PEFT-adapted HF model if adapters are present and HF available.
        adapter_dir = Path("models/biomedical_mistral_lora")
        if adapter_dir.exists() and HFModel is not None and PeftModel is not None:
            try:
                base_id = "mistralai/Mistral-7B-Instruct-v0.2"
                print("[llm] loading base model via Transformers for LoRA adapters …")
                
                device = "mps" if torch and torch.backends.mps.is_available() else "cpu"
                print(f"[llm] Using device: {device}")
                
                tok = AutoTokenizer.from_pretrained(base_id)
                mdl = HFModel.from_pretrained(
                    base_id, 
                    torch_dtype=torch.float16 if device == "mps" else torch.float32,
                    device_map="auto" if device != "cpu" else None
                )
                mdl = PeftModel.from_pretrained(mdl, adapter_dir)
                mdl.eval()
                mdl.tokenizer = tok  # attach for convenience
                
                # Move to device
                if device != "cpu":
                    mdl = mdl.to(device)
                    
                return mdl
            except Exception as exc:
                print(f"[llm] Warning: failed to load LoRA adapters → {exc}. Falling back to gguf model.")

    # 2) llama-cpp-python with optimized configuration
    if Llama is not None:
        if macbook_mode:
            print("[llm] loading via llama-cpp-python with MacBook optimization …")
            config = get_macbook_optimized_config()
            print(f"[llm] Using MacBook-optimized config: CPU-only, conservative memory")
        else:
            print("[llm] loading via llama-cpp-python with Metal acceleration …")
            config = get_optimal_config_for_model(str(selected_model))
        
        try:
            llm = Llama(model_path=str(selected_model), **config)
            if macbook_mode:
                print(f"[llm] ✅ MacBook CPU-only mode enabled - stable and memory-efficient")
            else:
                print(f"[llm] ✅ Standard mode enabled, using {config.get('n_gpu_layers', 0)} GPU layers")
            return llm
        except Exception as e:
            print(f"[llm] Loading failed: {e}")
            
            # Try with even more conservative settings
            print("[llm] Trying ultra-conservative fallback settings...")
            fallback_config = {
                "n_gpu_layers": 0,
                "n_ctx": 256,
                "n_batch": 1,
                "f16_kv": False,
                "use_mlock": False,
                "n_threads": 1,
                "verbose": False,
                "low_vram": True,
                "use_mmap": True,
            }
            
            try:
                return Llama(model_path=str(selected_model), **fallback_config)
            except Exception as e2:
                print(f"[llm] Fallback also failed: {e2}")
                # Try the smallest model as last resort
                if len(available_models) > 1:
                    smallest_model = min(available_models, key=lambda x: x.stat().st_size)
                    print(f"[llm] Trying smallest model as last resort: {smallest_model.name}")
                    try:
                        return Llama(model_path=str(smallest_model), **fallback_config)
                    except Exception as e3:
                        print(f"[llm] All attempts failed: {e3}")
                        raise
                else:
                    raise
    
    # 3) ctransformers fallback
    if AutoModelForCausalLM is not None:
        print("[llm] loading via ctransformers (CPU fallback) …")
        try:
            return AutoModelForCausalLM.from_pretrained(selected_model, model_type="llama")
        except Exception as e:
            print(f"[llm] ctransformers failed: {e}")
            raise
    
    raise RuntimeError("No LLM backend available - install llama-cpp-python or ctransformers") 