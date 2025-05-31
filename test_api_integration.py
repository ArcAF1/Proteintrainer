#!/usr/bin/env python3
"""
Test script for biomedical API integration
Shows how APIs enhance the AI's capabilities
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.biomedical_api_client import BiomedicalAPIClient
from src.api_enhanced_rag import APIEnhancedRAG
from src.rag_chat import enable_api_enhancement


async def test_api_client():
    """Test direct API access."""
    print("\n" + "="*60)
    print("🌐 TESTING BIOMEDICAL API CLIENT")
    print("="*60)
    
    async with BiomedicalAPIClient() as client:
        # Test 1: Search for creatine studies
        print("\n1. Searching PubMed for creatine studies...")
        pmids = await client.search_pubmed("creatine muscle performance", max_results=5)
        print(f"   Found {len(pmids)} articles: {pmids}")
        
        # Test 2: Get compound information
        print("\n2. Getting PubChem data for creatine...")
        compound = await client.get_compound_info("creatine")
        if compound:
            cid = compound.get("id", {}).get("id", {}).get("cid")
            print(f"   PubChem CID: {cid}")
            print(f"   Properties found: {len(compound.get('props', []))}")
        
        # Test 3: Search clinical trials
        print("\n3. Searching ClinicalTrials.gov...")
        trials = await client.search_clinical_trials("creatine supplementation", max_results=3)
        print(f"   Found {len(trials)} trials")
        for trial in trials:
            protocol = trial.get("protocolSection", {})
            nct_id = protocol.get("identificationModule", {}).get("nctId", "Unknown")
            title = protocol.get("identificationModule", {}).get("briefTitle", "Unknown")[:60]
            print(f"   - {nct_id}: {title}...")
            
        # Test 4: Search Europe PMC
        print("\n4. Searching Europe PMC for full-text articles...")
        papers = await client.search_europe_pmc("creatine absorption", max_results=3)
        print(f"   Found {len(papers)} papers")
        for paper in papers[:2]:
            print(f"   - {paper.get('title', 'Unknown')[:60]}...")


async def test_enhanced_rag():
    """Test API-enhanced RAG responses."""
    print("\n" + "="*60)
    print("🤖 TESTING API-ENHANCED RAG")
    print("="*60)
    
    # Create enhanced RAG
    rag = APIEnhancedRAG(use_apis=True)
    
    try:
        questions = [
            "What are the latest clinical trials on creatine?",
            "What is the molecular structure of creatine?",
            "What recent evidence supports creatine for muscle growth?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 40)
            
            # Get answer
            answer = await rag.answer(question)
            
            # Show first 500 chars of answer
            if len(answer) > 500:
                print(answer[:500] + "...\n[Answer truncated]")
            else:
                print(answer)
                
    finally:
        await rag.close()


async def compare_rag_modes():
    """Compare standard vs API-enhanced RAG."""
    print("\n" + "="*60)
    print("📊 COMPARING STANDARD VS API-ENHANCED RAG")
    print("="*60)
    
    question = "What are the latest findings on creatine absorption enhancement?"
    
    # Test without API enhancement
    print("\n🔴 Standard RAG (local data only):")
    print("-" * 40)
    enable_api_enhancement(False)
    from src.rag_chat import answer as rag_answer
    standard_answer = await rag_answer(question)
    print(standard_answer[:400] + "..." if len(standard_answer) > 400 else standard_answer)
    
    # Test with API enhancement
    print("\n\n🟢 API-Enhanced RAG (local + live data):")
    print("-" * 40)
    enable_api_enhancement(True)
    enhanced_answer = await rag_answer(question)
    print(enhanced_answer[:400] + "..." if len(enhanced_answer) > 400 else enhanced_answer)
    
    print("\n💡 Notice how API enhancement adds:")
    print("   - Recent clinical trial references (NCT IDs)")
    print("   - Latest paper citations (PMIDs)")
    print("   - Up-to-date findings beyond local data")


async def main():
    """Run all tests."""
    print("\n🧪 Biomedical API Integration Test Suite")
    print("This demonstrates how APIs enhance your AI\n")
    
    try:
        # Test 1: Direct API access
        await test_api_client()
        
        # Wait a bit between tests
        await asyncio.sleep(2)
        
        # Test 2: Enhanced RAG
        await test_enhanced_rag()
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Test 3: Compare modes
        await compare_rag_modes()
        
        print("\n\n✅ All tests complete!")
        
        print("\n📝 Summary:")
        print("1. APIs provide real-time access to:")
        print("   - 35M+ PubMed abstracts")
        print("   - 450K+ clinical trials")
        print("   - Chemical compound data")
        print("   - Medical terminology")
        
        print("\n2. API enhancement adds:")
        print("   - Current research (days old)")
        print("   - Evidence-based citations")
        print("   - Validated compound info")
        print("   - Trial recruitment status")
        
        print("\n3. To enable in GUI:")
        print("   - Open Advanced Tools")
        print("   - Toggle '🌐 Enable Live API Data'")
        print("   - Responses will include latest research!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting API integration tests...")
    print("This will make real API calls - ensure internet connection")
    asyncio.run(main()) 