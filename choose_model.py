#!/usr/bin/env python3
"""
Model selection helper for MacBook Pro 13-inch M1
Helps choose the optimal model based on available memory and performance needs.
"""

import psutil
from pathlib import Path

def get_available_models():
    """Get all available GGUF models with their sizes."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    models = []
    for model_path in models_dir.glob("*.gguf"):
        size_gb = model_path.stat().st_size / (1024**3)
        models.append({
            'name': model_path.name,
            'path': model_path,
            'size_gb': size_gb,
            'type': detect_model_type(model_path.name)
        })
    
    return sorted(models, key=lambda x: x['size_gb'])

def detect_model_type(filename):
    """Detect model type from filename."""
    filename_lower = filename.lower()
    if "medicine" in filename_lower:
        return "medicine-llm"
    elif "mistral" in filename_lower:
        return "mistral"
    elif "pmc" in filename_lower:
        return "pmc-llama"
    else:
        return "unknown"

def get_model_recommendation():
    """Get model recommendation based on system memory."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    print("🍎 MacBook Pro 13-inch M1 Model Recommendation")
    print("=" * 50)
    print(f"Available Memory: {available_gb:.1f} GB")
    print(f"Total Memory: {memory.total/(1024**3):.1f} GB")
    print(f"Memory Usage: {memory.percent:.1f}%")
    print()
    
    models = get_available_models()
    if not models:
        print("❌ No models found in the models directory!")
        return
    
    print("📋 Available Models:")
    print("-" * 30)
    
    recommended = None
    for i, model in enumerate(models, 1):
        size = model['size_gb']
        model_type = model['type']
        
        # Determine if model is suitable
        memory_needed = size * 1.5  # Model + overhead
        suitable = memory_needed <= available_gb
        
        status = "✅ RECOMMENDED" if suitable and recommended is None else "⚠️  HIGH MEMORY" if not suitable else "✅ OK"
        
        if suitable and recommended is None:
            recommended = model
        
        print(f"{i}. {model['name']}")
        print(f"   Size: {size:.1f} GB")
        print(f"   Type: {model_type}")
        print(f"   Memory needed: ~{memory_needed:.1f} GB")
        print(f"   Status: {status}")
        print()
    
    if recommended:
        print("🎯 RECOMMENDATION:")
        print(f"Use: {recommended['name']}")
        print(f"Size: {recommended['size_gb']:.1f} GB")
        print(f"Reason: Best balance of capability and memory usage for your MacBook")
        print()
        
        if recommended['type'] == 'mistral':
            print("💡 Performance Notes:")
            print("   • Mistral-7B is excellent for general biomedical tasks")
            print("   • Fast responses (5-15 seconds)")
            print("   • Leaves plenty of memory for other apps")
            print("   • Great for research, learning, and analysis")
        elif recommended['type'] == 'medicine-llm':
            print("💡 Performance Notes:")
            print("   • Medicine-LLM is specialized for medical tasks")
            print("   • Slower responses (15-30 seconds)")
            print("   • Uses more memory but more accurate for medical queries")
    else:
        print("⚠️  WARNING:")
        print("All models may be too large for optimal performance.")
        print("Consider closing other applications or using a smaller model.")
    
    print()
    print("🔧 To force using a specific model:")
    print("   1. Rename the unwanted models (add .backup extension)")
    print("   2. Or delete the larger models if you don't need them")
    print("   3. The system will automatically use the largest remaining model")

def main():
    get_model_recommendation()
    
    print()
    response = input("Would you like to see how to optimize your setup? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print()
        print("🚀 Optimization Tips for Your MacBook:")
        print("=" * 40)
        print("1. **Use Mistral-7B for daily work**")
        print("   • Rename medicine-llm model: medicine-llm-13b.Q4_K_M.gguf.backup")
        print("   • This forces the system to use the 4.1GB Mistral model")
        print()
        print("2. **Close memory-heavy apps before starting AI**")
        print("   • Close unused browser tabs")
        print("   • Quit video editing software")
        print("   • Close heavy IDEs if not needed")
        print()
        print("3. **Use the MacBook-optimized launcher**")
        print("   • ./start_optimized.command (automatically selects smaller models)")
        print("   • ./start_cpu_only.command (even more conservative)")
        print()
        print("4. **Monitor memory usage**")
        print("   • Activity Monitor > Memory tab")
        print("   • Keep memory pressure in green/yellow range")

if __name__ == "__main__":
    main() 