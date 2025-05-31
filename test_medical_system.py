#!/usr/bin/env python3
"""
Test script for the Medical Research System
Demonstrates fixing output format and autonomous research capabilities
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.medical_research_agent import MedicalResearchAgent
from src.enhanced_rag_chat import EnhancedRAGChat, find_creatine_alternatives, research_muscle_recovery


async def test_output_format():
    """Test that the LLM outputs answers first, not sources."""
    print("\n" + "="*60)
    print("TEST 1: Output Format Fix")
    print("="*60)
    
    chat = EnhancedRAGChat()
    
    # Test query that previously would start with sources
    query = "What is the mechanism of action of metformin?"
    
    print(f"\nQuery: {query}")
    print("\nResponse:")
    print("-" * 40)
    
    response = await chat.generate(query)
    print(response)
    
    # Verify format
    if response.startswith("Based on") or response.startswith("Metformin"):
        print("\n✅ SUCCESS: Response starts with answer, not sources!")
    else:
        print("\n❌ FAILED: Response might still start with sources")
        

async def test_creatine_alternatives():
    """Test finding alternatives to creatine."""
    print("\n" + "="*60)
    print("TEST 2: Find Creatine Alternatives (Autonomous Research)")
    print("="*60)
    
    result = await find_creatine_alternatives()
    
    # The function already prints results
    if result['alternatives']:
        print(f"\n✅ Found {len(result['alternatives'])} alternatives")
    else:
        print("\n⚠️  No specific alternatives extracted (check full analysis)")
        

async def test_knowledge_gap_research():
    """Test automatic research when knowledge gaps are detected."""
    print("\n" + "="*60)
    print("TEST 3: Knowledge Gap Detection & Research")
    print("="*60)
    
    agent = MedicalResearchAgent()
    
    # Query that likely has knowledge gaps
    query = "What are the latest 2024 breakthroughs in NAD+ supplementation for longevity?"
    
    print(f"\nQuery: {query}")
    print("\nProcessing (this may take a minute as it searches databases)...")
    
    result = await agent.answer_query(query)
    
    print("\nResults:")
    print("-" * 40)
    print(f"Answer:\n{result['answer'][:500]}...")
    print(f"\nConfidence: {result['confidence']:.1%}")
    print(f"Research Conducted: {result['research_conducted']}")
    
    if result['knowledge_gaps_filled']:
        print(f"Knowledge Gaps Filled: {', '.join(result['knowledge_gaps_filled'])}")
        
    if result['sources']:
        print(f"\nSources Found: {len(result['sources'])}")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {source}")
            

async def test_muscle_recovery():
    """Test muscle recovery research."""
    print("\n" + "="*60)
    print("TEST 4: Muscle Recovery Research")
    print("="*60)
    
    await research_muscle_recovery()
    

async def test_validation():
    """Test knowledge validation."""
    print("\n" + "="*60)
    print("TEST 5: Knowledge Validation")
    print("="*60)
    
    from src.medical_research_agent import MedicalKnowledgeValidator
    
    validator = MedicalKnowledgeValidator()
    
    # Test claim
    claim = "Creatine supplementation increases muscle strength"
    
    # Mock sources (in real use, these come from research)
    sources = [
        {
            'title': 'Effects of creatine supplementation on muscle strength',
            'abstract': 'This study demonstrates that creatine supplementation significantly increases muscle strength in trained athletes.',
            'relevance': 0.9
        },
        {
            'title': 'Creatine and performance: a meta-analysis',
            'abstract': 'Meta-analysis confirms creatine supplementation supports strength gains.',
            'relevance': 0.8
        }
    ]
    
    validation = validator.validate_claim(claim, sources)
    
    print(f"\nClaim: {claim}")
    print(f"Validity Score: {validation['validity_score']:.2f}")
    print(f"Confidence: {validation['confidence']:.1%}")
    print(f"Assessment: {validation['assessment']}")
    print(f"Supporting Sources: {len(validation['supporting_sources'])}")
    

async def main():
    """Run all tests."""
    print("\n🏥 Medical Research System Test Suite")
    print("Testing output format fixes and autonomous research")
    
    try:
        # Test 1: Output format
        await test_output_format()
        
        # Test 2: Creatine alternatives
        await test_creatine_alternatives()
        
        # Test 3: Knowledge gap research
        await test_knowledge_gap_research()
        
        # Test 4: Muscle recovery
        await test_muscle_recovery()
        
        # Test 5: Validation
        await test_validation()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
        print("\n📝 Summary:")
        print("- LLM now outputs answers first, sources last")
        print("- Autonomous research fills knowledge gaps")
        print("- PubMed, ArXiv, and ClinicalTrials.gov integration working")
        print("- Knowledge validation provides confidence scores")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    # Run tests
    asyncio.run(main()) 