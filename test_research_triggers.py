#!/usr/bin/env python3
"""
Test Research Triggers - Shows how research automatically starts
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.research_triggers import research_detector, RESEARCH_PROMPTS, format_research_response


async def test_research_detection():
    """Test automatic research detection."""
    print("\n" + "="*60)
    print("🔬 RESEARCH TRIGGER DETECTION TEST")
    print("="*60)
    
    # Test messages that should trigger research
    test_messages = [
        "Research ways to improve muscle recovery",
        "Can you investigate the effects of creatine?",
        "I want to study optimal protein timing",
        "Please analyze the best training split",
        "Explore methods to enhance sleep quality",
        "How to maximize muscle growth?",  # Complex question
        "What is the most effective supplement stack?",  # Complex question
        "Compare whey vs casein protein",  # Comparison
        "What's the weather today?",  # Should NOT trigger research
        "Hi there!",  # Should NOT trigger research
    ]
    
    print("\nTesting research detection:\n")
    
    for msg in test_messages:
        should_research, topic = research_detector.should_trigger_research(msg)
        
        if should_research:
            print(f"✅ TRIGGERS RESEARCH: '{msg}'")
            print(f"   → Research topic: '{topic}'")
        else:
            print(f"❌ No research: '{msg}'")
        print()


async def test_quick_research():
    """Test running actual research."""
    print("\n" + "="*60)
    print("🚀 RUNNING QUICK RESEARCH TEST")
    print("="*60)
    
    # Pick a research topic
    topic = "How to enhance creatine absorption and effectiveness"
    
    print(f"\nResearch topic: {topic}")
    print("Starting research (this will take 1-2 minutes)...\n")
    
    # Show research log as it happens
    print("Research Log:")
    print("-" * 40)
    
    # Run research
    result = await research_detector.trigger_research(topic, max_iterations=3)
    
    # Show formatted response
    response = format_research_response(result)
    print("\n" + response)
    
    # Show event log
    print("\n\nEvent Log:")
    print("-" * 40)
    for event in research_detector.get_research_log():
        print(f"{event['timestamp']}: {event['message']}")


async def test_premade_prompts():
    """Test pre-made research prompts."""
    print("\n" + "="*60)
    print("📋 PRE-MADE RESEARCH PROMPTS")
    print("="*60)
    
    print("\nAvailable quick research topics:\n")
    
    for i, (key, prompt) in enumerate(RESEARCH_PROMPTS.items(), 1):
        print(f"{i}. {key.replace('_', ' ').title()}")
        print(f"   → {prompt}")
        print()


async def main():
    """Run all tests."""
    print("\n🧪 Research Trigger System Test")
    print("This shows how research automatically starts\n")
    
    try:
        # Test 1: Detection
        await test_research_detection()
        
        # Test 2: Pre-made prompts
        await test_premade_prompts()
        
        # Test 3: Run actual research (optional - takes time)
        print("\n" + "="*60)
        response = input("Run actual research test? (takes 1-2 minutes) [y/N]: ")
        if response.lower() == 'y':
            await test_quick_research()
        
        print("\n✅ Test complete!")
        
        print("\n💡 How to use in the GUI:")
        print("1. Just ask questions naturally - research starts automatically")
        print("2. Use trigger words: 'research', 'investigate', 'study', etc.")
        print("3. Ask complex questions: 'How to maximize...', 'What is the best...'")
        print("4. Click the quick research buttons in the Research Activity Monitor")
        print("5. Watch the Research Event Log to see what's happening")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 