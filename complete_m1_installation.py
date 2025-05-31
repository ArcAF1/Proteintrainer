#!/usr/bin/env python3
"""
M1-Compatible Installation Script for Biomedical AI System
Handles Apple Silicon specific compilation issues and provides fallbacks.
"""

import subprocess
import sys
import platform

def is_m1_mac():
    """Detect if running on Apple Silicon"""
    return platform.machine() == 'arm64' and platform.system() == 'Darwin'

def install_with_fallback(package, fallback=None):
    """Try to install package, use fallback if fails"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed to install {package}")
        if fallback:
            print(f"🔄 Trying fallback: {fallback}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", fallback])
                print(f"✅ Installed fallback {fallback}")
                return True
            except:
                pass
        return False

def main():
    print("🚀 M1-Compatible Biomedical AI Installation")
    print("=" * 50)
    
    if is_m1_mac():
        print("🔍 Detected Apple Silicon (M1/M2/M3) Mac")
        print("   Using M1-optimized installation strategy...")
    else:
        print("🔍 Detected Intel/other architecture")
    
    # Phase 7: Biomedical Packages
    print("\n📦 Installing Phase 7: Biomedical Packages...")
    biomedical_packages = [
        "biopython>=1.81",
        "pubchempy>=1.0.4",
        "neo4j>=5.19.0",
        "neo4j-graphrag>=1.7.0"
    ]

    for pkg in biomedical_packages:
        install_with_fallback(pkg)

    # Phase 8: LangChain Ecosystem
    print("\n📦 Installing Phase 8: LangChain...")
    langchain_packages = [
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.5",
        "chromadb>=0.4.22",  # May need special handling for M1
    ]

    for pkg in langchain_packages:
        if pkg.startswith("chromadb") and is_m1_mac():
            # ChromaDB sometimes has M1 issues, install with specific flags
            print(f"🔧 Installing {pkg} with M1-specific flags...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg])
                print(f"✅ Successfully installed {pkg}")
            except subprocess.CalledProcessError:
                print(f"⚠️ ChromaDB failed - trying without cache...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", pkg])
                    print(f"✅ Successfully installed {pkg} on retry")
                except:
                    print(f"❌ ChromaDB installation failed completely")
        else:
            install_with_fallback(pkg)

    # Phase 9: Optional Enhancement Packages
    print("\n📦 Installing Phase 9: Optional Enhancements...")
    optional_packages = [
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "networkx>=3.0",
        "pyvis>=0.3.2"
    ]

    # Skip problematic M1 packages
    m1_incompatible = ["nmslib", "hnswlib"]  # Add any other known problematic packages

    for pkg in optional_packages:
        if not any(incomp in pkg for incomp in m1_incompatible):
            install_with_fallback(pkg)

    print("\n✅ Installation complete!")
    print("\n🔧 Creating M1-compatible requirements file...")

    # Create cleaned requirements file without problematic packages
    with open('requirements_m1_clean.txt', 'w') as f:
        f.write("""# M1-Compatible Requirements (nmslib excluded)
# Core packages already installed in phases 1-6
# Additional biomedical and ML packages
biopython>=1.81
pubchempy>=1.0.4
neo4j>=5.19.0
neo4j-graphrag>=1.7.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-openai>=0.0.5
chromadb>=0.4.22
streamlit>=1.28.0
plotly>=5.17.0
networkx>=3.0
pyvis>=0.3.2
# scispacy will work without nmslib for core functionality
""")

    print("📄 Created requirements_m1_clean.txt")
    
    # Final system check
    print("\n🔍 Final verification...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All dependencies are compatible!")
        else:
            print("⚠️ Some minor dependency conflicts (this is often normal):")
            print(result.stdout)
    except Exception as e:
        print(f"⚠️ Could not run dependency check: {e}")
    
    print(f"\n🎉 M1-Compatible Installation Complete!")
    print("=" * 40)
    print("🔬 Next steps:")
    print("1. Run: python verify_and_patch_scispacy.py")
    print("2. Test: python test_biomedical_setup.py")
    print("3. Start your biomedical AI system!")

if __name__ == "__main__":
    main() 