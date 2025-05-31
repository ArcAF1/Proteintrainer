#!/usr/bin/env python3
"""
Test the Experimental Research Engine
Demonstrates rapid hypothesis generation and testing
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.experimental_research_engine import ExperimentalResearchEngine, HypothesisEngine, integrate_with_graph


async def test_minimal_research():
    """Test the minimal research loop."""
    print("\n" + "="*60)
    print("🧪 EXPERIMENTAL RESEARCH ENGINE TEST")
    print("="*60)
    
    # Create engine
    engine = ExperimentalResearchEngine()
    engine = integrate_with_graph(engine)
    
    # Test questions
    questions = [
        "How to maximize muscle protein synthesis?",
        "What factors improve recovery between training sessions?",
        "Can we enhance creatine absorption?"
    ]
    
    for question in questions:
        print(f"\n📌 Research Question: {question}")
        print("-" * 40)
        
        # Run minimal iterations for quick test
        findings = await engine.research_loop(question, max_iterations=3)
        
        # Show summary
        summary = engine.get_summary()
        print(f"\n✅ Completed {summary['iterations']} iterations")
        print(f"📊 Generated {summary['total_innovations']} innovations")
        
        # Show top innovations
        if summary['top_innovations']:
            print("\n💡 Top Innovations:")
            for i, innovation in enumerate(summary['top_innovations'][:3], 1):
                print(f"   {i}. {innovation}")
                
        # Show a sample finding
        if findings:
            finding = findings[0]
            print(f"\n🔬 Sample Finding:")
            print(f"   Hypothesis: {finding['hypothesis']}")
            if finding.get('simulation'):
                print(f"   Simulation: {finding['simulation']['type']}")
                print(f"   Result: {finding['simulation'].get('recommendation', 'N/A')}")
                
        print("\n" + "="*60)


async def test_hypothesis_engine():
    """Test the hypothesis generation patterns."""
    print("\n🧠 Testing Hypothesis Engine")
    print("-" * 40)
    
    engine = HypothesisEngine()
    
    # Test gap-based generation
    gaps = ["mechanism of beta-alanine", "optimal rest periods", "protein timing"]
    hypotheses = await engine.generate_from_gaps([], gaps)
    
    print("Generated hypotheses from gaps:")
    for i, hyp in enumerate(hypotheses, 1):
        print(f"{i}. {hyp}")
        
    # Test connection finding
    concepts = ["muscle growth", "protein synthesis", "mTOR pathway", "training volume"]
    connections = engine.find_connections(concepts)
    
    print("\nFound connections:")
    for conn in connections:
        print(f"- {conn[0]} {conn[1]} {conn[2]}")


async def test_simulations():
    """Test the lightweight simulations."""
    print("\n📈 Testing Simulations")
    print("-" * 40)
    
    engine = ExperimentalResearchEngine()
    
    # Test different simulation types
    simulations = [
        ("High-intensity training improves VO2max", ["aerobic capacity"]),
        ("Creatine supplementation enhances strength", ["ATP", "phosphocreatine"]),
        ("Sleep quality affects recovery rate", ["muscle repair", "hormone release"])
    ]
    
    for hypothesis, mechanisms in simulations:
        print(f"\nHypothesis: {hypothesis}")
        result = await engine._run_simulation(hypothesis, mechanisms)
        
        if result:
            print(f"Simulation type: {result['type']}")
            print(f"Recommendation: {result['recommendation']}")
            
            # Show key metrics
            for key, value in result.items():
                if key not in ['type', 'recommendation'] and not key.startswith('_'):
                    print(f"  {key}: {value}")


async def main():
    """Run all tests."""
    print("\n🏃 Running Experimental Research Engine Tests")
    print("This demonstrates minimal overhead, maximum discovery approach")
    
    try:
        # Test 1: Minimal research loop
        await test_minimal_research()
        
        # Test 2: Hypothesis generation
        await test_hypothesis_engine()
        
        # Test 3: Simulations
        await test_simulations()
        
        print("\n✅ All tests completed!")
        print("\n💡 Key Benefits:")
        print("- No clinical trial bureaucracy")
        print("- Rapid hypothesis iteration")
        print("- Simple simulations for quick insights")
        print("- Focuses on your 11GB local knowledge")
        print("- Generates actionable innovations")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 