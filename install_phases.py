#!/usr/bin/env python3
"""
Phased installation script to avoid pip's resolution-too-deep error.
Installs packages in logical groups with compatible version constraints.
"""

import subprocess
import sys
import time

def run_pip_install(packages, description="", allow_failure=False):
    """Install packages with error handling."""
    print(f"\n🔧 {description}")
    print("=" * 50)
    print(f"Installing: {', '.join(packages)}")
    
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            # Only show the last few lines to avoid spam
            lines = result.stdout.strip().split('\n')
            if len(lines) > 3:
                print("...")
            for line in lines[-3:]:
                if line.strip():
                    print(f"   {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        
        if not allow_failure:
            print(f"💡 You may need to install these manually:")
            for pkg in packages:
                print(f"   pip install {pkg}")
        return False

def install_phase_1_core():
    """Install core foundational packages."""
    packages = [
        "packaging>=23.2,<24.0",
        "pyyaml",
        "python-dotenv",
        "markupsafe~=2.0",
        "requests",
        "tqdm",
        "rich>=13.0.0",
        "click>=8.0.0",
        "structlog",
    ]
    return run_pip_install(packages, "Phase 1: Core Utilities")

def install_phase_2_data():
    """Install data processing packages."""
    packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0", 
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "lxml",
        "beautifulsoup4",
        "feedparser",
    ]
    return run_pip_install(packages, "Phase 2: Data Processing")

def install_phase_3_ml_core():
    """Install core ML packages."""
    packages = [
        "torch>=2.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "einops>=0.7.0",
        "sentencepiece>=0.1.99",
    ]
    return run_pip_install(packages, "Phase 3: Core ML Frameworks")

def install_phase_4_specialized():
    """Install specialized ML packages."""
    packages = [
        "faiss-cpu",
        "sentence-transformers",
        "huggingface-hub>=0.20.0",
        "peft>=0.7.0",
        "evaluate>=0.4.0",
    ]
    return run_pip_install(packages, "Phase 4: Specialized ML")

def install_phase_5_nlp():
    """Install NLP packages."""
    packages = [
        "nltk>=3.8.0",
        "rouge-score>=0.1.0",
        "spacy>=3.7,<3.8",
        "scispacy",
    ]
    return run_pip_install(packages, "Phase 5: NLP Processing")

def install_phase_6_api():
    """Install API and web frameworks."""
    packages = [
        "pydantic==2.5.0",
        "pydantic-core==2.14.1", 
        "fastapi==0.104.1",
        "starlette==0.27.0",
        "gradio==4.21.0",
    ]
    return run_pip_install(packages, "Phase 6: API & Web Frameworks")

def install_phase_7_biomedical():
    """Install biomedical packages."""
    packages = [
        "biopython>=1.81",
        "pubchempy>=1.0.4",
        "neo4j==5.19.0",
        "neo4j-graphrag>=1.7.0",
    ]
    return run_pip_install(packages, "Phase 7: Biomedical Packages")

def install_phase_8_optional():
    """Install optional packages that may fail."""
    packages = [
        "ctransformers==0.2.27",
        "bitsandbytes>=0.41.0",
        "mlx>=0.0.6",
        "mlx-lm>=0.0.6",
        "llama-cpp-python",
        "memory-profiler>=0.61.0",
        "psutil",
        "GPUtil",
        "tensorboard>=2.14.0",
        "wandb>=0.15.0",
        "ragas>=0.0.18",
        "pytest",
    ]
    
    print(f"\n🔧 Phase 8: Optional Packages (may fail on some systems)")
    print("=" * 50)
    
    # Install these one by one since some may fail
    success_count = 0
    for package in packages:
        if run_pip_install([package], f"Installing {package}", allow_failure=True):
            success_count += 1
        time.sleep(1)  # Brief pause between installations
    
    print(f"\n✅ Successfully installed {success_count}/{len(packages)} optional packages")
    return True

def upgrade_langchain():
    """Upgrade LangChain packages to compatible versions."""
    packages = [
        "langchain-core>=0.3.0",
        "langchain-openai>=0.3.0", 
        "langchain-community>=0.3.0",
        "langchain>=0.3.0",
    ]
    return run_pip_install(packages, "LangChain Upgrade")

def main():
    print("🚀 Biomedical AI System - Phased Installation")
    print("=" * 60)
    print("This script installs packages in phases to avoid dependency conflicts.\n")
    
    phases = [
        ("Phase 1", install_phase_1_core),
        ("Phase 2", install_phase_2_data), 
        ("Phase 3", install_phase_3_ml_core),
        ("Phase 4", install_phase_4_specialized),
        ("Phase 5", install_phase_5_nlp),
        ("Phase 6", install_phase_6_api),
        ("Phase 7", install_phase_7_biomedical),
        ("LangChain", upgrade_langchain),
        ("Phase 8", install_phase_8_optional),
    ]
    
    successful_phases = 0
    
    for phase_name, phase_func in phases:
        print(f"\n🎯 Starting {phase_name}...")
        if phase_func():
            successful_phases += 1
            print(f"✅ {phase_name} completed successfully")
        else:
            print(f"⚠️  {phase_name} had issues - continuing anyway")
        
        # Brief pause between phases
        time.sleep(2)
    
    print(f"\n🎉 Installation Summary:")
    print("=" * 30)
    print(f"✅ Completed {successful_phases}/{len(phases)} phases successfully")
    
    # Final dependency check
    print(f"\n🔍 Final dependency check...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✅ All dependencies are compatible!")
        else:
            print("⚠️  Some dependency issues remain:")
            print(result.stdout)
    except Exception as e:
        print(f"❌ Could not check dependencies: {e}")
    
    print(f"\n🔬 Next Steps:")
    print("=" * 20)
    print("1. Run: python test_biomedical_setup.py")
    print("2. Check system status: python check_final_state.py") 
    print("3. Start your biomedical AI system!")

if __name__ == "__main__":
    main() 