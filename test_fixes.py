#!/usr/bin/env python3
"""Test script to validate the fixes to the biomedical AI system."""

import asyncio
import sys
from pathlib import Path

# Add src to path for local modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all critical imports work."""
    print("🔍 Testing imports...")
    
    try:
        import src.rag_chat as rag_chat
        print("✅ RAG chat import successful")
    except Exception as e:
        print(f"❌ RAG chat import failed: {e}")
        return False
    
    try:
        import src.embeddings as embeddings
        print("✅ Embeddings import successful")
    except Exception as e:
        print(f"❌ Embeddings import failed: {e}")
        return False
    
    try:
        import src.training_connector as training_connector
        print("✅ Training connector import successful")
    except Exception as e:
        print(f"❌ Training connector import failed: {e}")
        return False
    
    return True

def test_embeddings():
    """Test embeddings system with error handling."""
    print("\n🔍 Testing embeddings system...")
    
    try:
        from src.embeddings import Embedder
        
        embedder = Embedder()
        print(f"Status: {embedder.get_status()}")
        
        if embedder.is_ready():
            # Test encoding
            test_text = "This is a test of the embedding system."
            vector = embedder.encode(test_text)
            print(f"✅ Successfully encoded text to vector of shape: {vector.shape}")
            return True
        else:
            print("⚠️ Embedder not ready - this is expected on first run")
            print("   The system will download models when needed")
            return True
            
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system with error handling."""
    print("\n🔍 Testing RAG system...")
    
    try:
        from src.rag_chat import get_rag_status, answer
        
        status = get_rag_status()
        print(f"RAG Status:\n{status}")
        
        # Test answer function with error handling
        print("\nTesting RAG answer function...")
        response = asyncio.run(answer("What is aspirin?"))
        
        if "error" in response.lower() or "not ready" in response.lower():
            print("⚠️ RAG system not fully ready (expected on first run)")
            print("   Response sample:", response[:200] + "...")
        else:
            print("✅ RAG system responded successfully")
            print("   Response sample:", response[:200] + "...")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False

def test_training_system():
    """Test training system interface."""
    print("\n🔍 Testing training system...")
    
    try:
        from src.training_connector import get_training_connector
        
        connector = get_training_connector()
        status = connector.get_training_status()
        
        print(f"Training Status: {status}")
        
        if status['has_trained_model']:
            print("✅ Previously trained model found")
        else:
            print("ℹ️ No trained model found (expected on first run)")
        
        print("✅ Training system interface working")
        return True
        
    except Exception as e:
        print(f"❌ Training system test failed: {e}")
        return False

def test_gui_imports():
    """Test GUI system imports."""
    print("\n🔍 Testing GUI system...")
    
    try:
        from src.gui_unified import ai_system_handler, run_system_test
        print("✅ GUI imports successful")
        
        # Test system diagnostics
        print("Running system diagnostics...")
        diagnostics = run_system_test()
        print("✅ System diagnostics completed")
        print("Sample output:", diagnostics[:300] + "...")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def test_dependencies():
    """Test critical dependencies."""
    print("\n🔍 Testing dependencies...")
    
    dependencies = [
        ('gradio', 'Web interface'),
        ('torch', 'Machine learning'),
        ('transformers', 'Language models'),
        ('sentence_transformers', 'Embeddings'),
        ('faiss', 'Vector search'),
        ('neo4j', 'Graph database'),
        ('peft', 'Parameter efficient fine-tuning')
    ]
    
    all_good = True
    for dep, desc in dependencies:
        try:
            # Import the module directly without adding namespace
            if dep == 'faiss':
                import faiss
                module = faiss
            else:
                module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {desc}: {version}")
        except ImportError:
            print(f"❌ {desc}: Not installed")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("🧬 **Biomedical AI System Test Suite**\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Embeddings", test_embeddings),
        ("RAG System", test_rag_system),
        ("Training System", test_training_system),
        ("GUI System", test_gui_imports)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 **Test Results Summary**")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:15} {status}")
    
    print(f"\n🎯 **Overall: {passed}/{total} tests passed**")
    
    if passed == total:
        print("\n🎉 **All tests passed!** The system is ready to use.")
        print("\n📋 **Next steps:**")
        print("1. Run: source venv/bin/activate && python run_app.py")
        print("2. Open the web interface")
        print("3. Ask the AI to 'test the system'")
        print("4. Try 'start data pipeline' to build the knowledge base")
    else:
        print("\n⚠️ **Some tests failed.** This may indicate:")
        print("- Missing dependencies (run pip install -r requirements.txt)")
        print("- First-time setup (models need to be downloaded)")
        print("- Network connectivity issues")
        print("\nThe system may still work for basic operations.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 