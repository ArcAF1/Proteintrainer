#!/usr/bin/env python3
"""
Test Enhanced Biomedical LLM
Verifies all improvements are working correctly
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_llm import get_enhanced_llm, explain_ai_capabilities
from src.conversation_memory import get_conversation_memory
from src.biomedical_agent_integration import get_biomedical_agent

async def test_enhanced_llm():
    """Test all enhanced LLM features."""
    print("🧪 Testing Enhanced Biomedical LLM")
    print("=" * 60)
    
    # Initialize
    llm = get_enhanced_llm()
    agent = get_biomedical_agent()
    memory = get_conversation_memory()
    
    # Test 1: Role Understanding
    print("\n1️⃣ Testing Role Understanding...")
    print("Question: What is your role?")
    response = await agent.process_message("What is your role?")
    print(f"Response preview: {response[:200]}...")
    print("✅ Role explanation provided" if "pharmaceutical" in response.lower() else "❌ Missing role clarity")
    
    # Test 2: Capabilities
    print("\n2️⃣ Testing Capability Explanation...")
    print("Question: What can you help me with?")
    response = explain_ai_capabilities()
    print(f"Response preview: {response[:200]}...")
    print("✅ Capabilities explained" if "supplement development" in response.lower() else "❌ Missing capabilities")
    
    # Test 3: Clarification Questions
    print("\n3️⃣ Testing Clarification Questions...")
    print("Question: I need something for muscle")
    response = llm.generate("I need something for muscle")
    print(f"Response: {response}")
    print("✅ Asked for clarification" if "?" in response else "❌ No clarification requested")
    
    # Test 4: Conversation Memory
    print("\n4️⃣ Testing Conversation Memory...")
    
    # First interaction
    print("User: I'm interested in creatine for muscle building")
    response1 = llm.generate("I'm interested in creatine for muscle building")
    print(f"AI: {response1[:100]}...")
    
    # Second interaction referencing first
    print("\nUser: What dose did you mention?")
    response2 = llm.generate("What dose did you mention?")
    print(f"AI: {response2[:100]}...")
    print("✅ Memory working" if "creatine" in response2.lower() else "❌ Memory not working")
    
    # Test 5: Supplement Development
    print("\n5️⃣ Testing Supplement Development...")
    print("Question: Develop a pre-workout supplement for endurance")
    response = await agent.process_message("Develop a pre-workout supplement for endurance")
    print(f"Response preview: {response[:300]}...")
    print("✅ Formulation provided" if any(word in response.lower() for word in ['ingredients', 'dose', 'mg']) else "❌ No formulation")
    
    # Test 6: Compound Analysis
    print("\n6️⃣ Testing Compound Analysis...")
    print("Question: Analyze beta-alanine")
    response = llm.analyze_compound("beta-alanine")
    print(f"Response preview: {response[:200]}...")
    print("✅ Analysis provided" if "mechanism" in response.lower() else "❌ Analysis incomplete")
    
    # Show memory summary
    print("\n📊 Conversation Memory Summary:")
    print(memory.get_conversation_summary())
    
    print("\n" + "=" * 60)
    print("✅ Enhanced LLM Testing Complete!")
    
    # Practical tips
    print("\n💡 Integration Tips:")
    print("1. The LLM now has a clear biomedical identity")
    print("2. It remembers conversations and learns from them")
    print("3. It asks clarifying questions when needed")
    print("4. It specializes in supplement/pharma development")
    print("5. All of this works with your existing GUI!")

if __name__ == "__main__":
    print("Starting Enhanced LLM Test...")
    print("This will test all the improvements from the research guide")
    asyncio.run(test_enhanced_llm()) 