"""
Biomedical Agent Integration
Connects enhanced LLM with existing RAG and GUI systems
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from .enhanced_llm import get_enhanced_llm, explain_ai_capabilities
from .rag_chat import get_chat, enable_api_enhancement
from .conversation_memory import get_conversation_memory

logger = logging.getLogger(__name__)

class BiomedicalAgent:
    """Unified biomedical AI agent with enhanced capabilities."""
    
    def __init__(self):
        self.enhanced_llm = get_enhanced_llm()
        self.rag_chat = get_chat()
        self.memory = get_conversation_memory()
        
    async def process_message(self, message: str, history: List[List[str]] = None) -> str:
        """Process user message with enhanced biomedical focus."""
        
        message_lower = message.lower()
        
        # Check for capability questions
        if any(phrase in message_lower for phrase in [
            'what can you', 'what do you', 'who are you', 
            'your role', 'help me', 'capabilities'
        ]):
            # Use the enhanced explanation
            return explain_ai_capabilities()
        
        # Check for supplement development requests
        if any(phrase in message_lower for phrase in [
            'develop', 'formulate', 'create supplement', 'design supplement'
        ]):
            # Extract requirements from message
            requirements = self._extract_requirements(message)
            return self.enhanced_llm.develop_supplement(requirements)
        
        # Check for compound analysis
        if 'analyze' in message_lower or 'tell me about' in message_lower:
            compound = self._extract_compound_name(message)
            if compound:
                return self.enhanced_llm.analyze_compound(compound)
        
        # For general questions, use enhanced LLM with RAG
        try:
            # First try RAG for factual information
            if self.rag_chat.is_ready():
                rag_response = await self.rag_chat.answer(message)
                
                # Enhance with biomedical focus
                enhanced_prompt = f"""Based on this information from my knowledge base:

{rag_response}

Provide a response focused on pharmaceutical/supplement development implications. If relevant, mention:
- Formulation considerations
- Safety aspects
- Clinical applications
- Development opportunities

User question: {message}"""
                
                final_response = self.enhanced_llm.generate(enhanced_prompt, max_tokens=800)
                return final_response
            else:
                # Use enhanced LLM directly
                return self.enhanced_llm.generate(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Fallback to enhanced LLM
            return self.enhanced_llm.generate(message)
    
    def _extract_requirements(self, message: str) -> Dict[str, Any]:
        """Extract supplement development requirements from message."""
        requirements = {
            'goal': 'general health',
            'population': 'adults',
            'constraints': [],
            'preferences': []
        }
        
        # Simple extraction logic
        if 'muscle' in message.lower():
            requirements['goal'] = 'muscle building and recovery'
        elif 'energy' in message.lower():
            requirements['goal'] = 'energy and focus'
        elif 'sleep' in message.lower():
            requirements['goal'] = 'sleep quality'
        elif 'immune' in message.lower():
            requirements['goal'] = 'immune support'
        
        if 'natural' in message.lower():
            requirements['preferences'].append('natural ingredients preferred')
        if 'vegan' in message.lower():
            requirements['constraints'].append('vegan only')
        
        return requirements
    
    def _extract_compound_name(self, message: str) -> Optional[str]:
        """Extract compound name from message."""
        # Common compounds to look for
        compounds = [
            'creatine', 'caffeine', 'beta-alanine', 'citrulline',
            'ashwagandha', 'rhodiola', 'magnesium', 'zinc',
            'vitamin d', 'omega-3', 'melatonin', 'l-theanine'
        ]
        
        message_lower = message.lower()
        for compound in compounds:
            if compound in message_lower:
                return compound
        
        # Try to extract after "analyze" or "about"
        import re
        patterns = [
            r'analyze\s+(\w+)',
            r'about\s+(\w+)',
            r'what is\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1)
        
        return None
    
    def get_conversation_summary(self) -> str:
        """Get summary of current conversation."""
        return self.memory.get_conversation_summary()

# Global instance
_agent = None

def get_biomedical_agent() -> BiomedicalAgent:
    """Get or create the biomedical agent."""
    global _agent
    if _agent is None:
        _agent = BiomedicalAgent()
    return _agent

# Enhanced handler for GUI integration
async def enhanced_biomedical_handler(message: str, history: List[List[str]]) -> str:
    """Enhanced handler that integrates with existing GUI."""
    agent = get_biomedical_agent()
    
    # Process with enhanced capabilities
    response = await agent.process_message(message, history)
    
    # Add helpful context based on response
    if "I can help" in response or "capabilities" in response:
        response += "\n\n💡 **Quick Examples:**\n"
        response += "- 'Develop a pre-workout supplement for endurance athletes'\n"
        response += "- 'Analyze creatine for muscle building'\n"
        response += "- 'What are the latest studies on beta-alanine?'\n"
        response += "- 'Design a sleep support formula without melatonin'"
    
    return response

# Conversation memory functions for GUI
def save_conversation_to_memory(message: str, history: List[List[str]]) -> str:
    """Save important findings to memory."""
    memory = get_conversation_memory()
    
    if history and len(history) > 0:
        last_q, last_a = history[-1]
        
        # Check if this is important (contains compound info, dosages, etc)
        important_keywords = ['dose', 'mg', 'g/day', 'mechanism', 'interaction', 'contraindication']
        if any(keyword in last_a.lower() for keyword in important_keywords):
            memory.update_user_preferences('important_finding', f"{last_q}: {last_a[:200]}...")
            return "✅ **Saved to memory!** I'll remember this important information for future reference."
        else:
            return "💡 This conversation has been automatically saved. Mark specific findings as important by mentioning dosages or mechanisms."
    
    return "❌ No conversation to save yet."

def recall_from_memory(query: str) -> str:
    """Recall information from memory."""
    memory = get_conversation_memory()
    context = memory.get_relevant_context(query)
    
    if context:
        return f"📝 **From our previous conversations:**\n\n{context}"
    else:
        return "🤔 I don't have any saved conversations about that topic yet." 