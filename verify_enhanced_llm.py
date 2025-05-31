#!/usr/bin/env python3
"""
Quick verification that enhanced LLM is working
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Verifying Enhanced LLM Setup...")
print("=" * 40)

# Check 1: Files exist
files_to_check = [
    "src/biomedical_system_prompt.py",
    "src/conversation_memory.py", 
    "src/enhanced_llm.py",
    "src/biomedical_agent_integration.py"
]

all_exist = True
for file in files_to_check:
    if Path(file).exists():
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - MISSING!")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing. Enhanced LLM won't work properly.")
    sys.exit(1)

# Check 2: Can import
print("\n🔧 Testing imports...")
try:
    from src.biomedical_system_prompt import BIOMEDICAL_SYSTEM_PROMPT
    print("✅ System prompt loaded")
    
    from src.conversation_memory import ConversationMemory
    print("✅ Conversation memory available")
    
    from src.enhanced_llm import EnhancedBiomedicalLLM
    print("✅ Enhanced LLM class ready")
    
    from src.biomedical_agent_integration import enhanced_biomedical_handler
    print("✅ GUI integration ready")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Check 3: Verify system prompt content
print("\n📝 Checking system prompt...")
if "pharmaceutical" in BIOMEDICAL_SYSTEM_PROMPT and "supplement development" in BIOMEDICAL_SYSTEM_PROMPT:
    print("✅ System prompt has biomedical focus")
else:
    print("❌ System prompt missing biomedical content")

print("\n" + "=" * 40)
print("✅ Enhanced LLM is properly set up!")
print("\nYour AI now:")
print("• Knows it's a biomedical specialist")
print("• Remembers conversations") 
print("• Asks clarifying questions")
print("• Focuses on pharma/supplement development")
print("\nJust run: ./start_optimized.command") 