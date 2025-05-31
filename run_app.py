#!/usr/bin/env python3
"""Main entry point for the Biomedical AI Research Assistant.

Uses the new unified system architecture with comprehensive M1 optimization,
robust error handling, and automatic diagnostics.
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.unified_system import UnifiedBiomedicalSystem, run_diagnostics
    from src.knowledge_graph import wait_for_neo4j
    from src.config import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the project directory and dependencies are installed")
    sys.exit(1)


def check_neo4j():
    """Quick check if Neo4j is accessible."""
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
        
    except Exception:
        return False


def main():
    """Main application entry point with unified system."""
    print("🏥 Biomedical AI Research Assistant")
    print("=" * 50)
    
    # Wait for Neo4j to become healthy (up to 60 s)
    try:
        print("⏳ Waiting for Neo4j to start...")
        wait_for_neo4j(timeout=60)
        print("✅ Neo4j is ready")
    except Exception as exc:
        print(f"⚠️  Neo4j startup warning: {exc}")
        print("   (System will continue but knowledge graph features may be limited)")

    # Final connectivity check
    if not check_neo4j():
        print("⚠️  Neo4j connection check failed")
        print("   Knowledge graph features will be disabled")
        print("   To enable: docker-compose up -d")

    print("\n🔍 Running system diagnostics...")
    
    # Run diagnostics first
    try:
        diagnostics_results = run_diagnostics()
        
        # Count healthy components
        healthy_count = sum(1 for result in diagnostics_results.values() 
                           if result.get('healthy', False))
        total_count = len(diagnostics_results)
        
        if healthy_count < total_count * 0.7:  # At least 70% healthy
            print(f"\n⚠️  System health check: {healthy_count}/{total_count} components healthy")
            print("Some components may not work properly. Check the diagnostics above.")
            
            response = input("\nContinue anyway? (y/N): ").lower().strip()
            if response != 'y':
                print("Exiting. Fix the issues above and try again.")
                sys.exit(1)
        else:
            print(f"\n✅ System health check passed: {healthy_count}/{total_count} components healthy")
            
    except Exception as e:
        print(f"⚠️  Diagnostics failed: {e}")
        print("Continuing with limited health information...")

    print("\n🚀 Initializing unified biomedical system...")
    
    try:
        # Create and initialize the unified system
        system = UnifiedBiomedicalSystem()
        
        if not system.initialize():
            print("❌ System initialization failed")
            print("Check the logs and diagnostics above for details")
            sys.exit(1)
            
        print("✅ System initialized successfully")
        
        # Create and launch GUI
        print("🎨 Creating user interface…")
        system.create_gui('gradio')  # This function launches the GUI internally.

        # Inform the user and keep the process alive.
        print("\n🎉 System ready! The Gradio UI is now running.")
        print("Open your browser at the URL shown above and start chatting.")

    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("Check biomedical_system.log for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main() 