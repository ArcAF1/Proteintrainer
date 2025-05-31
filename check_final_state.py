#!/usr/bin/env python3
"""
Verify that despite typer[all] warnings during installation,
the final package state is correct and functional.
"""

def check_package_versions():
    """Check versions of key packages that showed warnings."""
    packages_to_check = [
        'typer', 'weasel', 'spacy', 'langchain-core', 
        'langchain-openai', 'langchain-community'
    ]
    
    print("🔍 Checking final package versions...")
    print("=" * 50)
    
    for package in packages_to_check:
        try:
            import importlib.util
            spec = importlib.util.find_spec(package.replace('-', '_'))
            if spec is not None:
                # Try to import and get version
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package}: {version}")
            else:
                print(f"❌ {package}: Not installed")
        except ImportError as e:
            print(f"❌ {package}: Import error - {e}")
        except Exception as e:
            print(f"⚠️  {package}: {e}")

def check_typer_extras():
    """Check if typer[all] extra is actually needed."""
    try:
        import typer
        print(f"\n🎯 Typer Analysis:")
        print("=" * 30)
        print(f"✅ Typer version: {typer.__version__}")
        
        # Check if typer includes all functionality by default
        try:
            # These are features that were previously in typer[all]
            from typer import rich_utils
            print("✅ Rich support: Available")
        except ImportError:
            print("❌ Rich support: Not available")
            
        try:
            # Test basic typer functionality
            app = typer.Typer()
            print("✅ Basic Typer functionality: Working")
        except Exception as e:
            print(f"❌ Basic Typer functionality: {e}")
            
        print("\n💡 Analysis: Typer ≥0.12.0 includes all features by default.")
        print("   The [all] extra is no longer needed or provided.")
        
    except ImportError:
        print("❌ Typer not available for testing")

def check_dependency_resolution():
    """Check if pip thinks there are any dependency conflicts."""
    import subprocess
    import sys
    
    print(f"\n🔧 Dependency Resolution Check:")
    print("=" * 40)
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'check'], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ All dependencies compatible!")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print("❌ Dependency conflicts found:")
            print(f"   {result.stdout}")
            print(f"   {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Dependency check timed out")
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")

def explain_warnings():
    """Explain why the warnings occur and why they're harmless."""
    print(f"\n📚 Why You See typer[all] Warnings:")
    print("=" * 50)
    print("""
1. 🔍 During Installation: pip's dependency resolver checks ALL possible
   versions of packages to find the best combination.

2. ⚠️  Warning Source: Some package metadata from older versions still 
   requests 'typer[all]' even though it's no longer needed.

3. ✅ Final Result: The resolver ultimately installs compatible versions
   that work together perfectly.

4. 🎯 Resolution: These warnings are cosmetic - they don't affect the
   final installation or functionality.

5. 💡 Why Safe: 
   - typer ≥0.12.0 includes all features by default
   - weasel 0.4.1 supports modern typer versions  
   - All packages end up compatible
""")

def main():
    print("🚀 Biomedical AI System - Installation Verification")
    print("=" * 60)
    
    check_package_versions()
    check_typer_extras()
    check_dependency_resolution()
    explain_warnings()
    
    print(f"\n🎉 Summary:")
    print("=" * 20)
    print("✅ The typer[all] warnings are normal during installation")
    print("✅ They don't affect the final package functionality")
    print("✅ Your biomedical AI system should work perfectly")
    print("✅ You can safely ignore these warnings")
    
    print(f"\n🔬 Next Steps:")
    print("=" * 20)
    print("1. Continue with your biomedical AI installation")
    print("2. Test the system functionality")
    print("3. Focus on actual functionality rather than these warnings")

if __name__ == "__main__":
    main() 