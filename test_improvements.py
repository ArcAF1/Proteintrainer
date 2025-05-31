#!/usr/bin/env python3
"""Test script for the improved biomedical LLM system.

Tests all the key improvements from the comprehensive fix:
- M1 optimization
- Robust RAG initialization
- System diagnostics
- Unified system architecture
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all new modules import correctly."""
    print("🧪 Testing imports...")
    
    try:
        from src.diagnostics import SystemDiagnostics, ErrorHandler, run_diagnostics
        print("  ✅ Diagnostics module")
        
        from src.unified_system import UnifiedBiomedicalSystem, TrainingSystemMediator
        print("  ✅ Unified system module")
        
        from src.embeddings import RobustRAGInitializer
        print("  ✅ Robust RAG initializer")
        
        from src.llm import get_optimal_m1_config
        print("  ✅ M1-optimized LLM loading")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_diagnostics():
    """Test system diagnostics."""
    print("\n🧪 Testing system diagnostics...")
    
    try:
        from src.diagnostics import SystemDiagnostics
        
        diagnostics = SystemDiagnostics()
        results = diagnostics.run_all_diagnostics()
        
        healthy_count = sum(1 for result in results.values() if result.get('healthy', False))
        total_count = len(results)
        
        print(f"  ✅ Diagnostics completed: {healthy_count}/{total_count} components healthy")
        return True
        
    except Exception as e:
        print(f"  ❌ Diagnostics failed: {e}")
        return False

def test_m1_optimization():
    """Test M1-specific optimizations."""
    print("\n🧪 Testing M1 optimizations...")
    
    try:
        from src.llm import get_optimal_m1_config
        import platform
        
        config = get_optimal_m1_config()
        is_m1 = platform.machine() == "arm64"
        
        print(f"  ✅ M1 config generated: {len(config)} parameters")
        print(f"  ℹ️  Running on: {platform.machine()}")
        print(f"  ℹ️  Metal optimized: {config['n_gpu_layers'] == -1 and is_m1}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ M1 optimization test failed: {e}")
        return False

def test_robust_rag():
    """Test robust RAG initialization."""
    print("\n🧪 Testing robust RAG initialization...")
    
    try:
        from src.embeddings import RobustRAGInitializer
        
        # Test with a fallback model that should work
        rag_init = RobustRAGInitializer("all-MiniLM-L6-v2")
        
        print(f"  ✅ RAG initializer created with dimension: {rag_init.dimension}")
        
        # Test document encoding
        test_docs = ["This is a test document.", "Another test document."]
        embeddings, valid_docs = rag_init.validate_and_encode(test_docs)
        
        print(f"  ✅ Document encoding: {len(valid_docs)} docs, {embeddings.shape} embeddings")
        
        # Test FAISS index creation
        index = rag_init.create_faiss_index(embeddings)
        print(f"  ✅ FAISS index created with {index.ntotal} vectors")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Robust RAG test failed: {e}")
        return False

def test_unified_system():
    """Test unified system architecture."""
    print("\n🧪 Testing unified system architecture...")
    
    try:
        from src.unified_system import UnifiedBiomedicalSystem
        
        # Create system without initializing
        system = UnifiedBiomedicalSystem()
        print(f"  ✅ Unified system created (initialized: {system.initialized})")
        
        # Test status before initialization
        status = system.get_status()
        print(f"  ✅ Status retrieved: {len(status)} keys")
        
        # Test direct chat (should handle uninitialized state gracefully)
        response = system.chat("Hello")
        print(f"  ✅ Chat response (uninitialized): {response[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Unified system test failed: {e}")
        return False

def test_error_handling():
    """Test enhanced error handling."""
    print("\n🧪 Testing error handling...")
    
    try:
        from src.diagnostics import ErrorHandler
        
        error_handler = ErrorHandler()
        
        # Test safe execution decorator
        @error_handler.safe_execute
        def test_function():
            return "success"
        
        result, error = test_function()
        print(f"  ✅ Safe execution (success): {result}, {error}")
        
        @error_handler.safe_execute
        def failing_function():
            raise ValueError("Test error")
        
        result, error = failing_function()
        print(f"  ✅ Safe execution (failure): {result is None}, {type(error).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Biomedical LLM System Improvements")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("M1 Optimization", test_m1_optimization),
        ("Diagnostics", test_diagnostics),
        ("Robust RAG", test_robust_rag),
        ("Unified System", test_unified_system),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System improvements are working correctly.")
        print("\n✅ You can now run the system with: python run_app.py")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the errors above.")
        print("\nSome improvements may not work as expected.")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 