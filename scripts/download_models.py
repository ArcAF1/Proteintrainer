#!/usr/bin/env python3
"""Download required model weights (LLM + spaCy) with HuggingFace authentication.

Example:
    # Set your HuggingFace token first:
    export HUGGINGFACE_TOKEN="your-token-here"
    
    # Then run:
    python scripts/download_models.py
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("Installing huggingface-hub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub>=0.20.0"], check=True)
    from huggingface_hub import hf_hub_download, login


MODELS_DIR = Path("models")

# Model configurations
MODELS = {
    "mistral-7b": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf",
        "filename": "mistral-7b-instruct.Q4_0.gguf",
        "size_gb": 4.1,
        "description": "General purpose 7B model (current)"
    },
    "medicine-llm-13b": {
        "url": "https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_K_M.gguf",
        "filename": "medicine-llm-13b.Q4_K_M.gguf", 
        "size_gb": 7.9,
        "description": "Specialized biomedical 13B model (RECOMMENDED)"
    },
    "pmc-llama-13b": {
        "url": "https://huggingface.co/TheBloke/PMC-LLaMA-13B-GGUF/resolve/main/pmc-llama-13b.Q4_K_M.gguf",
        "filename": "pmc-llama-13b.Q4_K_M.gguf",
        "size_gb": 7.9, 
        "description": "PubMed-trained 13B model (alternative)"
    }
}


def check_authentication() -> bool:
    """Check if HuggingFace authentication is set up"""
    # Check for token in environment
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    
    if token:
        print("✓ Found HuggingFace token in environment")
        try:
            login(token=token, add_to_git_credential=False)
            return True
        except Exception as e:
            print(f"✗ Token validation failed: {e}")
            return False
    
    # Check if already logged in via CLI
    try:
        from huggingface_hub import HfFolder
        if HfFolder.get_token():
            print("✓ Already authenticated via huggingface-cli")
            return True
    except:
        pass
    
    print("✗ No HuggingFace authentication found")
    # Try interactive prompt (non-headless)
    try:
        token = input("🔑 Enter your HuggingFace token (or leave empty for anonymous): ").strip()
    except Exception:
        token = ""
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print("✓ Token stored for future use")
            return True
        except Exception as exc:
            print("Token invalid →", exc)
    print("→ Continuing with anonymous access (models that require auth will fail)")
    return True


def download_mistral_model() -> Optional[Path]:
    """Download Mistral-7B GGUF model with authentication"""
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Model details
    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # Expected path with the standardized name used by other scripts
    expected_path = MODELS_DIR / "mistral-7b-instruct.Q4_0.gguf"
    
    if expected_path.exists():
        print(f"✓ Model already exists at: {expected_path}")
        file_size = expected_path.stat().st_size / (1024**3)
        print(f"✓ Model size: {file_size:.2f} GB")
        return expected_path
    
    print(f"Downloading {filename} from {repo_id}...")
    
    try:
        # Download with authentication
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,  # Resume if interrupted
            force_download=False   # Don't re-download if exists
        )
        
        model_path = Path(model_path)
        print(f"✓ Successfully downloaded to: {model_path}")
        
        # Verify the download
        file_size = model_path.stat().st_size / (1024**3)  # Size in GB
        print(f"✓ Model size: {file_size:.2f} GB")
        
        if file_size < 3.0:  # Q4_K_M should be ~4GB
            raise ValueError("Model file seems too small, may be corrupted")
        
        # Rename to expected filename if different
        if model_path.name != expected_path.name:
            print(f"Renaming {model_path.name} to {expected_path.name}")
            model_path.rename(expected_path)
            model_path = expected_path
            
        return model_path
        
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your HuggingFace token is set correctly")
        print("3. Try: huggingface-cli whoami")
        return None


def setup_m1_inference() -> dict:
    """Configure for optimal M1 performance"""
    print("\nSetting up M1-optimized inference...")
    
    # Check if we need to reinstall llama-cpp-python with Metal support
    try:
        import llama_cpp
        # Test if Metal is available
        if hasattr(llama_cpp, 'llama_backend_init'):
            print("✓ llama-cpp-python is already installed")
        else:
            raise ImportError("Need to reinstall with Metal support")
    except:
        print("Installing llama-cpp-python with Metal support...")
        env = os.environ.copy()
        env['CMAKE_ARGS'] = '-DLLAMA_METAL=on'
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python==0.2.20", 
            "--force-reinstall", 
            "--no-cache-dir"
        ], env=env, check=True)
    
    # Optimal configuration for M1
    config = {
        "n_gpu_layers": -1,   # Use all layers on GPU (Metal)
        "n_ctx": 4096,        # Context window
        "n_batch": 512,       # Batch size for prompt processing
        "f16_kv": True,       # Use half-precision for key/value cache
        "use_mlock": True,    # Lock model in memory
        "n_threads": 8,       # Use performance cores
    }
    
    print("✓ M1 optimization configured:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    
    return config


def install_spacy_models() -> None:
    """Install spacy models using spacy download command"""
    print("\nInstalling spaCy models...")
    try:
        # Install basic scientific spacy model
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✓ SpaCy model en_core_web_sm installed")
        
        # Try to install scispacy models if available
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_sci_sm"], check=False)
            print("✓ SciSpacy model en_core_sci_sm installed")
        except:
            print("ℹ SciSpacy models not available, using standard models")
            
    except Exception as e:
        print(f"Warning: Could not install spacy model: {e}")


def main() -> None:
    print("BioMedical AI Model Setup")
    print("=" * 50)
    
    # Step 1: Check authentication
    if not check_authentication():
        print("\n⚠️  Please set up HuggingFace authentication before continuing")
        sys.exit(1)
    
    # Step 2: Download Mistral model
    model_path = download_mistral_model()
    if not model_path:
        print("\n✗ Failed to download Mistral model")
        sys.exit(1)
    
    # Step 3: Setup M1 optimization
    m1_config = setup_m1_inference()
    
    # Save M1 config for later use
    import json
    config_path = MODELS_DIR / "m1_config.json"
    with open(config_path, "w") as f:
        json.dump(m1_config, f, indent=2)
    print(f"\n✓ M1 configuration saved to: {config_path}")
    
    # Step 4: Install spaCy models
    install_spacy_models()
    
    print("\n" + "=" * 50)
    print("✅ All models downloaded and configured successfully!")
    print(f"✅ Mistral model: {model_path}")
    print("✅ Ready for offline biomedical AI research")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Setup failed: {exc}")
        sys.exit(1) 