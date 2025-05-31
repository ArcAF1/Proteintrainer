#!/usr/bin/env python3
"""
Clean up any lingering processes from previous runs
Kills Python processes related to the biomedical system
"""
import os
import sys
import subprocess
import signal
import time

# Check if psutil is available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not installed. Using basic cleanup only.")
    print("   For full cleanup, run: pip3 install psutil")

def cleanup_processes():
    """Kill any lingering processes from previous runs."""
    print("🧹 Cleaning up previous processes...")
    
    if not PSUTIL_AVAILABLE:
        # Fallback to basic port cleanup
        basic_cleanup()
        return 0
    
    # Patterns to identify our processes
    process_patterns = [
        "gui_unified.py",
        "gradio",
        "proteintrainer",
        "biomedical",
        "rag_chat",
        "llama.cpp",
        "mistral-7b",
        "sentence-transformers"
    ]
    
    killed_count = 0
    
    # First, try to find and kill specific Python processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a Python process
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                # Check command line arguments
                cmdline = proc.info.get('cmdline', [])
                if cmdline:
                    cmdline_str = ' '.join(cmdline).lower()
                    
                    # Check if any of our patterns match
                    for pattern in process_patterns:
                        if pattern.lower() in cmdline_str:
                            # Don't kill ourselves!
                            if proc.pid != os.getpid():
                                print(f"  Killing process {proc.pid}: {' '.join(cmdline[:3])}...")
                                proc.terminate()
                                killed_count += 1
                                break
                                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Give processes time to terminate gracefully
    if killed_count > 0:
        time.sleep(2)
        
        # Force kill any remaining
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline:
                        cmdline_str = ' '.join(cmdline).lower()
                        for pattern in process_patterns:
                            if pattern.lower() in cmdline_str and proc.pid != os.getpid():
                                print(f"  Force killing stubborn process {proc.pid}")
                                proc.kill()
                                break
            except:
                pass
    
    # Also clean up any orphaned llama.cpp processes
    try:
        # Find processes using significant memory that might be LLMs
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                # Look for processes using more than 2GB RAM
                if proc.memory_info().rss > 2 * 1024 * 1024 * 1024:  # 2GB
                    name = proc.name().lower()
                    if any(x in name for x in ['llama', 'ggml', 'mistral', 'metal']):
                        print(f"  Killing high-memory process {proc.pid} ({name})")
                        proc.terminate()
                        killed_count += 1
            except:
                pass
    except:
        pass
    
    # Clean up any stale lock files
    lock_files = [
        "active_config.lock",
        ".llm_instance.lock",
        "research_active.lock"
    ]
    
    for lock_file in lock_files:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"  Removed lock file: {lock_file}")
            except:
                pass
    
    # Clear any shared memory segments (macOS specific)
    if sys.platform == "darwin":
        try:
            # Clear shared memory
            subprocess.run(["ipcrm", "-a"], capture_output=True, text=True)
        except:
            pass
    
    if killed_count > 0:
        print(f"✅ Cleaned up {killed_count} processes")
    else:
        print("✅ No lingering processes found")
    
    # Final safety delay
    time.sleep(1)
    
    return killed_count

def basic_cleanup():
    """Basic cleanup without psutil."""
    # Try to kill processes using lsof (macOS/Linux)
    if sys.platform in ["darwin", "linux"]:
        ports = [7860, 7861, 7862, 7863, 7864, 7865]
        for port in ports:
            try:
                # Find process using port
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    pid = result.stdout.strip()
                    print(f"  Killing process on port {port} (pid: {pid})")
                    subprocess.run(["kill", "-9", pid])
            except:
                pass
    
    # Clean up lock files
    lock_files = [
        "active_config.lock",
        ".llm_instance.lock", 
        "research_active.lock"
    ]
    
    for lock_file in lock_files:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"  Removed lock file: {lock_file}")
            except:
                pass

def cleanup_ports():
    """Free up ports that might be in use."""
    print("\n🔌 Checking ports...")
    
    if not PSUTIL_AVAILABLE:
        print("  Skipping port check (psutil not available)")
        return
    
    ports_to_check = [7860, 7861, 7862, 7863, 7864, 7865]
    freed_count = 0
    
    for port in ports_to_check:
        try:
            # Find process using this port
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    try:
                        proc = psutil.Process(conn.pid)
                        if proc.pid != os.getpid():  # Don't kill ourselves
                            print(f"  Freeing port {port} (pid: {conn.pid})")
                            proc.terminate()
                            freed_count += 1
                    except:
                        pass
        except:
            pass
    
    if freed_count > 0:
        print(f"✅ Freed {freed_count} ports")
        time.sleep(2)
    else:
        print("✅ All ports are free")

if __name__ == "__main__":
    print("🚀 Biomedical AI System - Cleanup Utility")
    print("=" * 50)
    
    # Run cleanup
    cleanup_processes()
    cleanup_ports()
    
    print("\n✅ Cleanup complete! Safe to start the application.")
    print("\nRun: ./start_optimized.command") 