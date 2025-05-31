#!/usr/bin/env python3
"""
Test script to verify LLM can handle longer contexts after configuration updates
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.rag_chat import answer

async def test_longer_responses():
    """Test that the LLM can generate longer responses about metformin."""
    
    print("🧪 Testing longer response generation...")
    print("=" * 60)
    
    # Test questions that should generate detailed responses
    test_questions = [
        "What is the mechanism of action of metformin? Please provide a detailed explanation.",
        "Explain how metformin works for type 2 diabetes, including its effects on glucose metabolism, insulin sensitivity, and cellular pathways.",
        "What are the primary molecular targets and pathways affected by metformin in treating diabetes?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("-" * 60)
        
        try:
            response = await answer(question)
            
            # Check response length
            char_count = len(response)
            word_count = len(response.split())
            
            print(f"✅ Response generated successfully!")
            print(f"📊 Statistics:")
            print(f"   • Characters: {char_count}")
            print(f"   • Words: {word_count}")
            print(f"   • Meets 1800 char requirement: {'Yes' if char_count >= 1800 else 'No'}")
            
            print(f"\n📄 Response preview (first 500 chars):")
            print(response[:500] + "..." if len(response) > 500 else response)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
            # Check if it's a token limit error
            if "exceed context window" in str(e):
                print("⚠️  Still getting context window errors!")
                print("   Please restart the application for config changes to take effect.")
    
    print("\n" + "=" * 60)
    print("💡 Note: If you're still getting context window errors:")
    print("   1. Restart the application (config is loaded at startup)")
    print("   2. Check that macbook_config.json has n_ctx=2048")
    print("   3. Ensure the LLM is using the updated configuration")

if __name__ == "__main__":
    print("🔬 LLM Longer Context Test")
    print("Testing ability to generate responses with at least 1800 characters")
    print("")
    
    asyncio.run(test_longer_responses()) 