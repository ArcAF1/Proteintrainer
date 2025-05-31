#!/usr/bin/env python3
"""
Test and install neo4j-graphrag if missing.
This script can be run independently to fix the neo4j-graphrag issue.
"""

import sys
import subprocess
import platform

def is_m1_mac():
    """Detect if running on Apple Silicon"""
    return platform.machine() == 'arm64' and platform.system() == 'Darwin'

def test_import():
    """Test if neo4j-graphrag can be imported"""
    try:
        import neo4j_graphrag
        print("✅ neo4j-graphrag is available")
        print(f"   Version: {getattr(neo4j_graphrag, '__version__', 'unknown')}")
        return True
    except ImportError as e:
        print(f"❌ neo4j-graphrag import failed: {e}")
        return False

def install_neo4j_graphrag():
    """Install neo4j-graphrag with proper error handling"""
    print("🔧 Installing neo4j-graphrag...")
    
    # First try standard installation
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "neo4j>=5.19.0", "neo4j-graphrag>=1.7.0"])
        print("✅ Standard installation successful")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Standard installation failed")
    
    # For M1 Macs, try with specific flags
    if is_m1_mac():
        print("🔧 Trying M1-specific installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "neo4j>=5.19.0", "neo4j-graphrag>=1.7.0"])
            print("✅ M1-specific installation successful")
            return True
        except subprocess.CalledProcessError:
            print("⚠️ M1-specific installation failed")
    
    # Try upgrading pip first
    print("🔧 Upgrading pip and trying again...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "neo4j>=5.19.0", "neo4j-graphrag>=1.7.0"])
        print("✅ Installation successful after pip upgrade")
        return True
    except subprocess.CalledProcessError:
        print("❌ All installation attempts failed")
        return False

def main():
    print("🔍 Testing neo4j-graphrag installation...")
    print("=" * 40)
    
    if is_m1_mac():
        print("🔍 Detected Apple Silicon Mac")
    else:
        print("🔍 Detected Intel/other architecture")
    
    # Test if already working
    if test_import():
        print("\n🎉 neo4j-graphrag is working correctly!")
        return
    
    # Try to install
    print("\n🔧 Attempting to install neo4j-graphrag...")
    if install_neo4j_graphrag():
        print("\n🔍 Testing installation...")
        if test_import():
            print("\n🎉 Installation successful!")
        else:
            print("\n❌ Installation completed but import still fails")
    else:
        print("\n❌ Installation failed")
        print("\nTroubleshooting steps:")
        print("1. Check internet connection")
        print("2. Try: pip install --upgrade pip")
        print("3. Try: python complete_m1_installation.py")
        print("4. Check Python version compatibility")

if __name__ == "__main__":
    main() 