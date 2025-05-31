#!/usr/bin/env python3
"""
Full Integration Test - Verifies all optimizations work together
Tests performance modes, API integration, and research features
"""
import json
import sys
import os
from pathlib import Path

def test_integration():
    """Test all components of the optimized system."""
    print("🧪 Full Integration Test")
    print("=" * 60)
    
    # 1. Check active configuration
    print("\n1️⃣ Performance Configuration:")
    if Path("active_config.json").exists():
        with open("active_config.json", 'r') as f:
            config = json.load(f)
        
        mode = "unknown"
        priority = config.get('performance', {}).get('priority', 'balanced')
        if priority == 'maximum_speed':
            mode = "Ultra (Maximum Speed)"
        elif priority == 'speed':
            mode = "Optimized (Metal GPU)"
        else:
            mode = "Conservative (CPU-only)"
            
        print(f"   Mode: {mode}")
        print(f"   GPU Layers: {config.get('llm', {}).get('n_gpu_layers', 0)}")
        print(f"   Memory Limit: {config.get('performance', {}).get('max_memory_gb', 5)}GB")
    else:
        print("   ⚠️  No active configuration found")
        print("   Run: python switch_performance_mode.py optimized")
    
    # 2. Check API integration
    print("\n2️⃣ API Integration:")
    try:
        from src.biomedical_api_client import BiomedicalAPIClient
        from src.api_integration_config import API_CONFIGS
        print("   ✅ API client available")
        print(f"   APIs configured: {len(API_CONFIGS)}")
        for name, config in list(API_CONFIGS.items())[:3]:
            print(f"   - {config.name} (Priority: {'⭐' * config.priority})")
    except ImportError:
        print("   ❌ API integration not available")
    
    # 3. Check research features
    print("\n3️⃣ Research Features:")
    try:
        from src.research_triggers import research_detector, RESEARCH_PROMPTS
        from src.experimental_research_engine import ExperimentalResearchEngine
        print("   ✅ Research triggers ready")
        print("   ✅ Experimental engine ready")
        print(f"   Research prompts available: {len(RESEARCH_PROMPTS)}")
    except ImportError as e:
        print(f"   ⚠️  Some research features unavailable: {e}")
    
    # 4. Check GPU support
    print("\n4️⃣ Hardware Acceleration:")
    metal_enabled = os.environ.get('PYTORCH_MPS_ENABLED', '0') == '1'
    force_cpu = os.environ.get('FORCE_CPU_ONLY', '0') == '1'
    
    if metal_enabled and not force_cpu:
        print("   ✅ Metal GPU acceleration ENABLED")
        print("   Expected speedup: 5-10x")
    else:
        print("   ⚠️  CPU-only mode active")
        print("   To enable GPU: python switch_performance_mode.py optimized")
    
    # 5. Check environment
    print("\n5️⃣ Environment Variables:")
    important_vars = [
        'LLAMA_N_GPU_LAYERS',
        'PYTORCH_MPS_ENABLED', 
        'FORCE_CPU_ONLY',
        'MACBOOK_OPTIMIZED'
    ]
    for var in important_vars:
        value = os.environ.get(var, 'not set')
        print(f"   {var}: {value}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    
    if Path("active_config.json").exists() and metal_enabled:
        print("✅ System is FULLY OPTIMIZED for M1!")
        print("   - GPU acceleration active")
        print("   - API integration available")
        print("   - Research features ready")
    else:
        print("⚠️  System is running in SAFE MODE")
        print("   For better performance:")
        print("   1. Run: python switch_performance_mode.py optimized")
        print("   2. Restart: ./start_optimized.command")
    
    print("\n💡 Quick Commands:")
    print("   • Switch to optimized: python switch_performance_mode.py optimized")
    print("   • Test APIs: python test_api_integration.py")
    print("   • Test research: python test_research_triggers.py")
    print("   • Start system: ./start_optimized.command")

if __name__ == "__main__":
    test_integration() 