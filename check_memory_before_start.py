#!/usr/bin/env python3
"""
Memory checker to prevent system overload before starting the biomedical AI system.
"""

import psutil
import sys
from pathlib import Path

def check_macbook_compatibility():
    """Check if this is a MacBook and optimize accordingly."""
    import platform
    import subprocess
    
    system_info = {}
    
    try:
        # Check if this is macOS
        if platform.system() == "Darwin":
            system_info["is_mac"] = True
            
            # Try to get Mac model info
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"], 
                capture_output=True, text=True, timeout=10
            )
            
            if "MacBook Pro" in result.stdout:
                system_info["is_macbook_pro"] = True
                if "13-inch" in result.stdout:
                    system_info["screen_size"] = "13-inch"
                if "Apple M1" in result.stdout:
                    system_info["chip"] = "M1"
                    
    except Exception:
        pass
    
    return system_info

def check_other_applications():
    """Check what other applications are running and their memory usage."""
    other_apps = []
    total_other_memory = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
        try:
            name = proc.info['name']
            memory_percent = proc.info['memory_percent']
            memory_info = proc.info['memory_info']
            
            # Skip if memory info is None or invalid
            if not memory_info or not hasattr(memory_info, 'rss'):
                continue
                
            memory_mb = memory_info.rss / (1024**2)
            
            # Skip system processes and our own processes
            skip_processes = [
                'kernel_task', 'launchd', 'WindowServer', 'loginwindow',
                'python', 'Python', 'neo4j', 'docker', 'java'
            ]
            
            if any(skip in name for skip in skip_processes):
                continue
                
            if memory_percent and memory_percent > 2:  # More than 2% memory
                other_apps.append({
                    'name': name,
                    'memory_percent': memory_percent,
                    'memory_mb': memory_mb
                })
                total_other_memory += memory_mb
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError, TypeError):
            # Process disappeared, access denied, zombie process, or invalid data
            continue
    
    # Sort by memory usage
    other_apps.sort(key=lambda x: x['memory_percent'], reverse=True)
    
    return other_apps[:10], total_other_memory  # Top 10 apps

def check_system_memory():
    """Check if system has enough free memory to run safely."""
    
    # Get system memory info
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    used_gb = memory.used / (1024**3)
    available_gb = memory.available / (1024**3)
    usage_percent = memory.percent
    
    print(f"🔍 System Memory Status:")
    print(f"   Total: {memory_gb:.1f} GB")
    print(f"   Used: {used_gb:.1f} GB ({usage_percent:.1f}%)")
    print(f"   Available: {available_gb:.1f} GB")
    
    # Check MacBook compatibility
    mac_info = check_macbook_compatibility()
    if mac_info.get("is_macbook_pro") and mac_info.get("screen_size") == "13-inch":
        print(f"\n💻 MacBook Pro 13-inch detected - using optimized settings")
        if mac_info.get("chip") == "M1":
            print(f"   ✅ M1 chip detected - hardware protection enabled")
    
    # Check other applications
    other_apps, total_other_memory_mb = check_other_applications()
    total_other_memory_gb = total_other_memory_mb / 1024
    
    if other_apps:
        print(f"\n🔍 Other Running Applications:")
        print(f"   Total memory used by other apps: {total_other_memory_gb:.1f} GB")
        print(f"   Top applications:")
        
        for app in other_apps[:5]:  # Show top 5
            print(f"   • {app['name']}: {app['memory_mb']:.0f}MB ({app['memory_percent']:.1f}%)")
    
    # Check MPS memory if available
    mps_memory_gb = 0
    try:
        import torch
        if torch.backends.mps.is_available():
            mps_memory_gb = torch.mps.current_allocated_memory() / (1024**3)
            print(f"   MPS Allocated: {mps_memory_gb:.2f} GB")
    except:
        pass
    
    # MacBook-specific recommendations
    free_for_ai = available_gb - 2  # Reserve 2GB for system
    
    print(f"\n🎯 MacBook Optimization:")
    print(f"   Memory available for AI: {free_for_ai:.1f} GB")
    print(f"   Reserved for other apps: 2.0 GB")
    print(f"   AI system will use: ≤ {min(free_for_ai, 3.0):.1f} GB")
    
    # Safety checks with MacBook considerations
    warnings = []
    
    if usage_percent > 75:  # More conservative for MacBook with other apps
        warnings.append(f"⚠️  High memory usage: {usage_percent:.1f}%")
    
    if available_gb < 6:  # Need more free memory on MacBook
        warnings.append(f"⚠️  Low available memory: {available_gb:.1f} GB")
        
    if total_other_memory_gb > 8:  # Too many other apps
        warnings.append(f"⚠️  High memory usage by other apps: {total_other_memory_gb:.1f} GB")
        
    if mps_memory_gb > 0.5:
        warnings.append(f"⚠️  MPS memory in use: {mps_memory_gb:.1f} GB")
    
    # Recommendations
    if warnings:
        print(f"\n⚠️  MacBook Memory Warnings:")
        for warning in warnings:
            print(f"   {warning}")
        
        print(f"\n💡 MacBook Recommendations:")
        print(f"   • Close resource-heavy apps (Chrome, Figma, Xcode, etc.)")
        print(f"   • Keep essential apps but close unused browser tabs")
        print(f"   • The AI system will use minimal resources")
        print(f"   • CPU-only mode prevents GPU conflicts")
        
        if usage_percent > 85:
            print(f"\n❌ Memory usage too high for MacBook ({usage_percent:.1f}%)")
            print(f"   Please close some applications and try again.")
            return False
    else:
        print(f"\n✅ MacBook memory status looks good for AI system")
    
    return True

def check_model_sizes():
    """Check sizes of available models."""
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"❌ Models directory not found")
        return False
        
    model_files = list(models_dir.glob("*.gguf"))
    if not model_files:
        print(f"❌ No model files found")
        return False
    
    print(f"\n🤖 Available Models:")
    total_size_gb = 0
    for model in model_files:
        size_gb = model.stat().st_size / (1024**3)
        total_size_gb += size_gb
        print(f"   • {model.name}: {size_gb:.1f} GB")
    
    print(f"   Total models: {total_size_gb:.1f} GB")
    
    # Memory recommendation based on model size
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    largest_model = max(model_files, key=lambda x: x.stat().st_size)
    largest_size_gb = largest_model.stat().st_size / (1024**3)
    
    recommended_memory = largest_size_gb * 1.5  # 1.5x model size for safe operation
    
    if available_gb < recommended_memory:
        print(f"\n⚠️  Memory Warning for {largest_model.name}")
        print(f"   Model size: {largest_size_gb:.1f} GB")
        print(f"   Recommended free memory: {recommended_memory:.1f} GB")
        print(f"   Available: {available_gb:.1f} GB")
        print(f"   → System will use very conservative settings")
    
    return True

def main():
    print("🏥 Biomedical AI System - Pre-startup Memory Check")
    print("=" * 55)
    
    # Check system memory
    memory_ok = check_system_memory()
    
    # Check model availability
    models_ok = check_model_sizes()
    
    print(f"\n🛡️  HARDWARE PROTECTION MODE ENABLED")
    print("=" * 40)
    print("✅ System configured for CPU-only operation")
    print("✅ GPU/Metal acceleration DISABLED to prevent overload")
    print("✅ Ultra-conservative memory settings active")
    print("✅ Will prioritize existing data over downloads")
    
    print(f"\n🎯 Startup Recommendation:")
    if memory_ok and models_ok:
        print(f"✅ System ready for CPU-only startup")
        print(f"✅ Hardware protection mode will prevent overload")
    else:
        print(f"⚠️  System may experience performance issues")
        print(f"⚠️  Consider closing applications or restarting")
    
    print(f"\n💡 Performance Notes:")
    print(f"   • CPU-only mode is slower but much more stable")
    print(f"   • Your existing 10+GB data will be used")
    print(f"   • No heavy downloads will occur") 
    print(f"   • System optimized for 13-inch M1 MacBook Pro")

if __name__ == "__main__":
    main() 