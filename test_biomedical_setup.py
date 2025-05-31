#!/usr/bin/env python3
"""Test biomedical package installations."""

def test_rdkit():
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CCO')
        print(f"✅ RDKit: Ethanol has {mol.GetNumAtoms()} atoms")
        return True
    except ImportError:
        print("❌ RDKit not installed - Run: conda install -c conda-forge rdkit")
        return False

def test_neo4j_graphrag():
    try:
        import neo4j_graphrag
        print(f"✅ Neo4j GraphRAG: Version {neo4j_graphrag.__version__ if hasattr(neo4j_graphrag, '__version__') else 'available'}")
        return True
    except ImportError:
        print("❌ Neo4j GraphRAG not installed - Should be available from requirements.txt")
        return False

def test_biopython():
    try:
        from Bio import SeqIO
        import Bio
        print(f"✅ BioPython: Version {Bio.__version__}")
        return True
    except ImportError:
        print("❌ BioPython not installed - Should be available from requirements.txt")
        return False

def test_scispacy():
    try:
        import scispacy
        import spacy
        print(f"✅ SciSpaCy: Available with spaCy {spacy.__version__}")
        return True
    except ImportError:
        print("❌ SciSpaCy not installed - Should be available from requirements.txt")
        return False

def test_pubchempy():
    try:
        import pubchempy as pcp
        # Simple test query
        print("✅ PubChemPy: Available for chemical data access")
        return True
    except ImportError:
        print("❌ PubChemPy not installed - Should be available from requirements.txt")
        return False

def test_faiss():
    try:
        import faiss
        print("✅ FAISS: Available for vector search")
        return True
    except ImportError:
        print("❌ FAISS not installed - Should be available from requirements.txt")
        return False

def test_transformers():
    try:
        import transformers
        print(f"✅ Transformers: Version {transformers.__version__}")
        return True
    except ImportError:
        print("❌ Transformers not installed - Should be available from requirements.txt")
        return False

def test_mlx_m1():
    try:
        import mlx.core as mx
        print("✅ MLX: Available for M1/M2 acceleration")
        return True
    except ImportError:
        print("❌ MLX not installed - M1/M2 acceleration unavailable")
        return False

def test_core_system():
    """Test core system modules."""
    try:
        import sys
        sys.path.append('src')
        from src.diagnostics import SystemDiagnostics
        print("✅ Core System: Diagnostics module available")
        return True
    except ImportError as e:
        print(f"❌ Core System: Import error - {e}")
        return False

if __name__ == "__main__":
    print("🧬 Testing Biomedical AI System Package Setup")
    print("=" * 50)
    
    tests = [
        ("Core System", test_core_system),
        ("Neo4j GraphRAG", test_neo4j_graphrag),
        ("BioPython", test_biopython),
        ("PubChemPy", test_pubchempy),
        ("SciSpaCy", test_scispacy),
        ("FAISS", test_faiss),
        ("Transformers", test_transformers),
        ("RDKit", test_rdkit),
        ("MLX (M1/M2)", test_mlx_m1),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n🔬 Testing {name}...")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} packages working")
    
    if failed == 0:
        print("🎉 All biomedical packages are ready!")
        print("💡 System ready for biomedical AI research")
    else:
        print(f"⚠️  {failed} packages need attention")
        print("📖 See BIOMEDICAL_SETUP.md for installation guides")
    
    print("\n🚀 Next steps:")
    print("1. Run: ./installation.command (if not done)")
    print("2. Run: ./start.command (to launch system)")
    print("3. Test: 'What is the mechanism of aspirin?'") 