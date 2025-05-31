#!/usr/bin/env python3
"""
Install Neo4j GraphRAG with biomedical-optimized dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command with error handling."""
    print(f"[install] {description}")
    print(f"[cmd] {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    print("🧬 Installing Neo4j GraphRAG for Biomedical AI")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment!")
        print("   Recommend running: source venv/bin/activate")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print(f"✅ Python: {sys.version}")
    print(f"✅ Virtual env: {sys.prefix}")
    
    # Installation steps
    installations = [
        # Core GraphRAG package
        ("pip install neo4j-graphrag", "Installing core Neo4j GraphRAG"),
        
        # Enhanced biomedical support 
        ("pip install 'neo4j-graphrag[sentence-transformers]'", "Adding sentence-transformers support"),
        ("pip install 'neo4j-graphrag[experimental]'", "Adding experimental features"),
        
        # Optional: LLM providers (comment out if not needed)
        # ("pip install 'neo4j-graphrag[openai]'", "Adding OpenAI support"),
        # ("pip install 'neo4j-graphrag[anthropic]'", "Adding Anthropic support"),
        ("pip install 'neo4j-graphrag[ollama]'", "Adding Ollama support (for local LLMs)"),
        
        # Biomedical enhancements
        ("pip install biopython>=1.81", "Installing BioPython for biological sequences"),
        ("pip install pubchempy>=1.0.4", "Installing PubChemPy for chemical data"),
        ("pip install rdkit || echo 'RDKit installation skipped - install via conda if needed'", "Attempting RDKit installation"),
    ]
    
    success_count = 0
    
    for cmd, description in installations:
        if run_command(cmd, description):
            success_count += 1
        else:
            print(f"⚠️  Continuing despite error in: {description}")
    
    print("\n" + "=" * 50)
    print(f"📊 Installation Results: {success_count}/{len(installations)} successful")
    
    # Verification
    print("\n🔬 Verifying installation...")
    
    verification_tests = [
        ("import neo4j_graphrag", "Neo4j GraphRAG core"),
        ("from neo4j_graphrag.embeddings import OpenAIEmbeddings", "GraphRAG embeddings"),
        ("from neo4j_graphrag.retrievers import VectorRetriever", "GraphRAG retrievers"),
        ("from neo4j_graphrag.generation import GraphRAG", "GraphRAG generation"),
        ("import Bio; print(f'BioPython {Bio.__version__}')", "BioPython"),
        ("import pubchempy; print('PubChemPy OK')", "PubChemPy"),
    ]
    
    working_packages = 0
    
    for test_code, package_name in verification_tests:
        try:
            exec(test_code)
            print(f"✅ {package_name}: Working")
            working_packages += 1
        except Exception as e:
            print(f"❌ {package_name}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Final Status: {working_packages}/{len(verification_tests)} packages working")
    
    if working_packages >= len(verification_tests) - 1:  # Allow 1 failure
        print("🎉 Neo4j GraphRAG installation successful!")
        print("\n📋 What's installed:")
        print("  • neo4j-graphrag: Graph-enhanced RAG capabilities")
        print("  • sentence-transformers: Advanced embeddings")
        print("  • experimental: Latest GraphRAG features")
        print("  • ollama: Local LLM support")
        print("  • biopython: Biological sequence analysis")
        print("  • pubchempy: Chemical compound data")
        print("  • rdkit: Chemical informatics (if available)")
        
        print("\n🚀 Next steps:")
        print("1. Ensure Neo4j is running: docker compose up -d")
        print("2. Test the system: python test_biomedical_setup.py")
        print("3. Run diagnostics: python -c 'from src.diagnostics import SystemDiagnostics; SystemDiagnostics().run_all_checks()'")
        
    else:
        print("⚠️  Some packages failed to install")
        print("📖 See BIOMEDICAL_SETUP.md for manual installation guides")
        print("🔧 For RDKit, run: conda install -c conda-forge rdkit")
    
    print("\n💡 Usage example:")
    print("""
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j import GraphDatabase

# Initialize embeddings optimized for biomedical text
embedder = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')

# Connect to Neo4j and create retriever
driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))
retriever = VectorRetriever(driver, "biomedical_index", embedder)
""")

if __name__ == "__main__":
    main() 