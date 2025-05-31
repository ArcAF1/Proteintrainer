#!/usr/bin/env python3
"""
Fix the repetitive typer[all] warnings during pip installation.

The issue: Some dependency is requesting typer[all] but typer>=0.12.0 
no longer provides the [all] extra (it's included by default).
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """Run a command with error handling."""
    print(f"[fix] {description}")
    print(f"[cmd] {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    print("🔧 Fixing typer[all] warnings")
    print("=" * 40)
    
    # Check current typer version
    print("\n1. Checking current typer version...")
    if not run_command("pip show typer", "Getting typer info"):
        print("❌ Could not check typer version")
        return False
        
    # Upgrade typer to a version without [all] extra
    print("\n2. Upgrading typer to fix [all] extra warnings...")
    if not run_command("pip install 'typer>=0.12.1' --upgrade", "Upgrading typer"):
        print("❌ Could not upgrade typer")
        return False
        
    # Verify the fix
    print("\n3. Verifying typer version...")
    if not run_command("pip show typer", "Checking new typer version"):
        print("❌ Could not verify typer version")
        return False
        
    print("\n✅ typer warnings should now be fixed!")
    print("\n💡 Note: typer>=0.12.1 includes all functionality by default")
    print("   No need for [all] extra anymore!")

if __name__ == "__main__":
    main() 