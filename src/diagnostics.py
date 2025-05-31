"""System diagnostics for biomedical LLM system.

Comprehensive health checks and troubleshooting for M1 Mac deployment.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandler:
    """Enhanced error handling with diagnostics."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('biomedical_system.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def safe_execute(self, func: Callable) -> Callable:
        """Decorator for safe function execution with detailed error reporting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                self.logger.info(f"✅ {func.__name__} executed successfully")
                return result, None
            except Exception as e:
                self.logger.error(f"❌ {func.__name__} failed: {e}", exc_info=True)
                return None, e
        return wrapper


class SystemDiagnostics:
    """Comprehensive system health checks for M1 Mac biomedical LLM system."""
    
    def __init__(self):
        self.results = {}
        
    def run_all_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Run complete system diagnostics."""
        print("🔍 Running comprehensive system diagnostics...")
        
        checks = [
            ("platform", self._check_platform),
            ("python_env", self._check_python_environment),
            ("llama_cpp", self._check_llama_cpp),
            ("metal_acceleration", self._check_metal_acceleration),
            ("faiss", self._check_faiss),
            ("embeddings", self._check_embeddings),
            ("neo4j", self._check_neo4j),
            ("disk_space", self._check_disk_space),
            ("memory", self._check_memory),
            ("models", self._check_models),
        ]
        
        for name, check_func in checks:
            try:
                self.results[name] = check_func()
            except Exception as e:
                self.results[name] = {
                    'healthy': False,
                    'error': str(e),
                    'recommendation': 'Fix the underlying error and re-run diagnostics'
                }
        
        self._print_summary()
        return self.results
        
    def _check_platform(self) -> Dict[str, Any]:
        """Check platform compatibility."""
        arch = platform.machine()
        os_version = platform.platform()
        python_version = sys.version
        
        is_m1 = arch == "arm64"
        is_macos = platform.system() == "Darwin"
        
        return {
            'healthy': is_m1 and is_macos,
            'architecture': arch,
            'os_version': os_version,
            'python_version': python_version,
            'recommendation': 'System optimized for M1 Mac' if is_m1 and is_macos 
                            else 'Some optimizations may not work on non-M1 systems'
        }
        
    def _check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment setup."""
        venv_path = Path("venv/bin/python")
        current_python = sys.executable
        
        is_venv = venv_path.exists()
        
        # Improved virtual environment detection
        using_venv = (
            "venv" in current_python or
            os.environ.get("VIRTUAL_ENV") is not None or
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        return {
            'healthy': is_venv and using_venv,
            'venv_exists': is_venv,
            'using_venv': using_venv,
            'python_path': current_python,
            'virtual_env': os.environ.get("VIRTUAL_ENV", "Not set"),
            'recommendation': 'Activate virtual environment: source venv/bin/activate' 
                            if not using_venv else 'Python environment OK'
        }
        
    def _check_llama_cpp(self) -> Dict[str, Any]:
        """Check llama-cpp-python installation and Metal support."""
        try:
            from llama_cpp import Llama
            
            # Try to check compilation flags
            try:
                # Create a minimal test instance
                test_model_path = Path("models/mistral-7b-instruct-v0.2.q4_0.gguf")
                if test_model_path.exists():
                    llm = Llama(model_path=str(test_model_path), n_ctx=512, n_gpu_layers=1, verbose=False)
                    has_metal = True
                else:
                    has_metal = "Unknown (no test model)"
            except Exception:
                has_metal = False
                
            return {
                'healthy': True,
                'installed': True,
                'metal_support': has_metal,
                'recommendation': 'llama-cpp-python ready' if has_metal 
                                else 'Consider reinstalling with Metal support'
            }
        except ImportError:
            return {
                'healthy': False,
                'installed': False,
                'error': 'llama-cpp-python not installed',
                'recommendation': 'Install with: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal'
            }
            
    def _check_metal_acceleration(self) -> Dict[str, Any]:
        """Check Metal Performance Shaders availability."""
        try:
            import torch
            
            mps_available = torch.backends.mps.is_available()
            mps_built = torch.backends.mps.is_built()
            
            # Check for Xcode
            try:
                result = subprocess.run(['xcode-select', '-p'], capture_output=True, text=True)
                xcode_path = result.stdout.strip()
                has_xcode = "Xcode.app" in xcode_path
            except:
                has_xcode = False
                
            return {
                'healthy': mps_available and mps_built,
                'mps_available': mps_available,
                'mps_built': mps_built,
                'xcode_installed': has_xcode,
                'xcode_path': xcode_path if has_xcode else None,
                'recommendation': 'Metal acceleration ready' if (mps_available and mps_built)
                                else 'Install full Xcode from App Store for Metal support'
            }
        except ImportError:
            return {
                'healthy': False,
                'error': 'PyTorch not installed',
                'recommendation': 'Install PyTorch: pip install torch'
            }
            
    def _check_faiss(self) -> Dict[str, Any]:
        """Check FAISS installation and functionality."""
        try:
            import faiss
            
            # Test basic functionality
            test_dim = 384
            test_index = faiss.IndexFlatIP(test_dim)
            
            return {
                'healthy': True,
                'version': faiss.__version__ if hasattr(faiss, '__version__') else 'Unknown',
                'cpu_support': True,
                'recommendation': 'FAISS ready for vector operations'
            }
        except ImportError:
            return {
                'healthy': False,
                'error': 'FAISS not installed',
                'recommendation': 'Install FAISS: pip install faiss-cpu'
            }
            
    def _check_embeddings(self) -> Dict[str, Any]:
        """Check sentence-transformers and embedding models."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load a small model on CPU to avoid MPS memory conflicts
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                test_embedding = model.encode(["test"])
                dimension = test_embedding.shape[1]
                
                return {
                    'healthy': True,
                    'model_loaded': True,
                    'test_dimension': dimension,
                    'recommendation': 'Embedding system ready'
                }
            except Exception as e:
                return {
                    'healthy': False,
                    'model_loaded': False,
                    'error': str(e),
                    'recommendation': 'Check internet connection or use offline models'
                }
                
        except ImportError:
            return {
                'healthy': False,
                'error': 'sentence-transformers not installed',
                'recommendation': 'Install sentence-transformers: pip install sentence-transformers'
            }
            
    def _check_neo4j(self) -> Dict[str, Any]:
        """Check Neo4j connectivity."""
        try:
            from neo4j import GraphDatabase
            from src.config import settings
            
            try:
                driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password)
                )
                
                with driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    
                driver.close()
                
                return {
                    'healthy': True,
                    'connected': True,
                    'uri': settings.neo4j_uri,
                    'recommendation': 'Neo4j connection successful'
                }
                
            except Exception as e:
                return {
                    'healthy': False,
                    'connected': False,
                    'error': str(e),
                    'recommendation': 'Start Neo4j: docker-compose up -d'
                }
                
        except ImportError:
            return {
                'healthy': False,
                'error': 'neo4j driver not installed',
                'recommendation': 'Install neo4j: pip install neo4j'
            }
            
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            sufficient = free_gb > 20  # Need at least 20GB for models and data
            
            return {
                'healthy': sufficient,
                'free_gb': round(free_gb, 2),
                'total_gb': round(total_gb, 2),
                'recommendation': 'Sufficient disk space' if sufficient 
                                else f'Warning: Only {free_gb:.1f}GB free. Need 20GB+ for full operation'
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'recommendation': 'Could not check disk space'
            }
            
    def _check_memory(self) -> Dict[str, Any]:
        """Check system memory."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            sufficient = available_gb > 2  # Need at least 2GB for M1-optimized LLM
            
            return {
                'healthy': sufficient,
                'available_gb': round(available_gb, 2),
                'total_gb': round(total_gb, 2),
                'recommendation': 'Sufficient memory for M1-optimized system' if sufficient 
                                else f'Warning: Only {available_gb:.1f}GB RAM available. May need to close other apps for optimal performance'
            }
        except ImportError:
            return {
                'healthy': True,  # Don't fail if psutil not available
                'error': 'psutil not installed',
                'recommendation': 'Install psutil for memory monitoring: pip install psutil'
            }
            
    def _check_models(self) -> Dict[str, Any]:
        """Check for required model files."""
        models_dir = Path("models")
        models_found = []
        
        if not models_dir.exists():
            return {
                'healthy': False,
                'models_found': [],
                'models_dir_exists': False,
                'recommendation': 'Create models directory and download base model: python scripts/download_models.py'
            }
        
        # Look for any GGUF model files
        gguf_files = list(models_dir.glob("*.gguf"))
        model_files = [f.name for f in gguf_files]
        
        # Check for LoRA adapters
        lora_dirs = [d.name for d in models_dir.iterdir() if d.is_dir() and "lora" in d.name.lower()]
        
        has_base_model = len(gguf_files) > 0
        
        models_found = model_files + lora_dirs
        
        return {
            'healthy': has_base_model,
            'models_found': models_found,
            'models_dir_exists': True,
            'base_models': model_files,
            'lora_adapters': lora_dirs,
            'recommendation': f'Base model found: {model_files[0]}' if has_base_model 
                            else 'Download base model: python scripts/download_models.py'
        }
        
    def _print_summary(self):
        """Print diagnostic summary."""
        print("\n" + "="*60)
        print("🏥 SYSTEM HEALTH SUMMARY")
        print("="*60)
        
        healthy_count = sum(1 for result in self.results.values() if result.get('healthy', False))
        total_count = len(self.results)
        
        for component, result in self.results.items():
            status = "✅" if result.get('healthy', False) else "❌"
            print(f"{status} {component.replace('_', ' ').title()}: {result.get('recommendation', 'No recommendation')}")
            
            if not result.get('healthy', False) and 'error' in result:
                print(f"   Error: {result['error']}")
                
        print(f"\nOverall Health: {healthy_count}/{total_count} components healthy")
        
        if healthy_count == total_count:
            print("🎉 System is ready for operation!")
        else:
            print("⚠️  Some issues need attention before full operation.")
            
        print("="*60)


def run_diagnostics():
    """Convenience function to run all diagnostics."""
    diagnostics = SystemDiagnostics()
    return diagnostics.run_all_diagnostics()


if __name__ == "__main__":
    run_diagnostics() 