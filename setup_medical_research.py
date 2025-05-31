#!/usr/bin/env python3
"""
Setup script for the Medical Research System
Installs dependencies and configures the system
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n📦 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} - Success!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e}")
        return False
    return True


def main():
    print("🏥 Medical Research System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version}")
    
    # Create directories
    print("\n📁 Creating directories...")
    dirs = [
        "medical_research",
        "medical_research/chroma_db",
        "medical_research/logs",
        "medical_research/library"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Core requirements
    if not run_command(
        "pip install -r requirements_medical.txt",
        "Installing medical research dependencies"
    ):
        print("\n⚠️  Some dependencies failed. Trying individual installs...")
        
        deps = [
            "chromadb",
            "arxiv",
            "requests",
            "spacy",
            "scispacy",
            "scikit-learn",
            "aiohttp",
            "pandas",
            "lxml"
        ]
        
        for dep in deps:
            run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Download spaCy model
    run_command(
        "python -m spacy download en_core_sci_sm",
        "Downloading medical NLP model"
    )
    
    # Create config file
    print("\n⚙️  Creating configuration...")
    config = {
        "mistral_prompt_format": "answer_first",
        "research_auto_trigger": True,
        "knowledge_coverage_threshold": 0.7,
        "max_research_sources": 10,
        "confidence_display": True,
        "pubmed_retmax": 10,
        "arxiv_max_results": 5,
        "clinical_trials_max": 5
    }
    
    import json
    with open("medical_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✅ Configuration saved to medical_config.json")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import chromadb
        print("  ✅ ChromaDB")
    except:
        print("  ❌ ChromaDB - Please install manually: pip install chromadb")
        
    try:
        import arxiv
        print("  ✅ ArXiv API")
    except:
        print("  ❌ ArXiv API - Please install manually: pip install arxiv")
        
    try:
        import spacy
        print("  ✅ spaCy")
    except:
        print("  ❌ spaCy - Please install manually: pip install spacy")
    
    print("\n✨ Setup complete!")
    print("\n📝 Next steps:")
    print("1. Run: python test_medical_system.py")
    print("2. Start GUI: python -m src.gui_unified")
    print("3. Ask medical questions to trigger autonomous research")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 