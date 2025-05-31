#!/usr/bin/env python3
"""
Switch between performance modes for your biomedical AI system.
"""
import shutil
import sys
from pathlib import Path


def switch_mode(mode: str):
    """Switch between conservative and optimized modes."""
    configs = {
        'conservative': 'macbook_config.json',
        'optimized': 'm1_optimized_config.json',
        'ultra': 'm1_ultra_config.json'
    }
    
    if mode not in configs:
        print(f"❌ Invalid mode: {mode}")
        print("Available modes: conservative, optimized, ultra")
        return False
        
    source = Path(configs[mode])
    target = Path('active_config.json')
    
    if not source.exists():
        print(f"❌ Config file not found: {source}")
        return False
        
    # Backup current config
    if target.exists():
        backup = Path('active_config.backup.json')
        shutil.copy(target, backup)
        print(f"📦 Backed up current config to {backup}")
    
    # Copy new config
    shutil.copy(source, target)
    print(f"✅ Switched to {mode} mode")
    
    # Show mode details
    mode_info = {
        'conservative': """
🐌 Conservative Mode:
- CPU only (no GPU)
- Low memory usage (5GB max)
- Slower but very stable
- Good for multitasking
""",
        'optimized': """
🚀 Optimized Mode:
- Metal GPU acceleration (24 layers)
- Higher memory usage (10GB max)
- 5-10x faster inference
- Balanced for M1 MacBook Pro
""",
        'ultra': """
⚡ Ultra Mode:
- Maximum GPU usage
- Highest memory allocation
- Fastest possible speed
- For dedicated AI work only
"""
    }
    
    print(mode_info.get(mode, ""))
    return True


def benchmark_current_mode():
    """Run a quick benchmark of current mode."""
    print("\n📊 Running benchmark...")
    
    try:
        from src.m1_optimized_llm import benchmark_inference
        results = benchmark_inference("What are the mechanisms of creatine in muscle growth?")
        
        print(f"""
Benchmark Results:
- Average time: {results['avg_time']:.2f}s
- Tokens/second: {results['tokens_per_second']:.1f}
- Min time: {results['min_time']:.2f}s
- Max time: {results['max_time']:.2f}s
""")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def main():
    """Main entry point."""
    print("🎛️  Performance Mode Switcher")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("""
Usage: python switch_performance_mode.py [mode]

Modes:
  conservative - Safe mode for multitasking (CPU only)
  optimized    - Balanced M1 performance (recommended)
  ultra        - Maximum speed (dedicated use)
  
Example:
  python switch_performance_mode.py optimized
  
Current mode can be checked in active_config.json
""")
        return
        
    mode = sys.argv[1].lower()
    
    if mode == 'benchmark':
        benchmark_current_mode()
    else:
        success = switch_mode(mode)
        if success and input("\nRun benchmark? [y/N]: ").lower() == 'y':
            benchmark_current_mode()


if __name__ == "__main__":
    main() 