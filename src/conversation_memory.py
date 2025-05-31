"""
Conversation Memory System
Maintains context across interactions and learns from conversations
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib

class ConversationMemory:
    """Simple but effective conversation memory system."""
    
    def __init__(self, memory_dir: str = "conversation_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Current session memory (fast access)
        self.current_session = {
            'id': self._generate_session_id(),
            'start_time': datetime.now().isoformat(),
            'interactions': [],
            'learned_facts': {},
            'user_preferences': {}
        }
        
        # Load persistent memory
        self.persistent_memory = self._load_persistent_memory()
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def _load_persistent_memory(self) -> Dict:
        """Load memory from previous sessions."""
        memory_file = self.memory_dir / "persistent_memory.json"
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                return json.load(f)
        return {
            'important_facts': {},
            'user_projects': {},
            'compound_history': [],
            'preferences': {}
        }
    
    def add_interaction(self, user_message: str, ai_response: str, metadata: Dict = None):
        """Store a conversation turn."""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'assistant': ai_response,
            'metadata': metadata or {}
        }
        
        # Add to current session
        self.current_session['interactions'].append(interaction)
        
        # Extract important information
        self._extract_and_store_facts(user_message, ai_response)
        
        # Save periodically (every 5 interactions)
        if len(self.current_session['interactions']) % 5 == 0:
            self._save_session()
    
    def _extract_and_store_facts(self, user_msg: str, ai_response: str):
        """Extract important facts from conversation."""
        # Look for compound names, dosages, goals
        keywords = {
            'compounds': ['creatine', 'beta-alanine', 'citrulline', 'caffeine'],
            'dosages': ['mg', 'g', 'dose', 'daily'],
            'goals': ['muscle', 'endurance', 'recovery', 'strength'],
            'projects': ['developing', 'formulating', 'designing']
        }
        
        text = (user_msg + " " + ai_response).lower()
        
        # Extract mentioned compounds
        for compound in keywords['compounds']:
            if compound in text:
                if compound not in self.current_session['learned_facts']:
                    self.current_session['learned_facts'][compound] = []
                
                # Store context about this compound
                if 'dose' in text or 'mg' in text or 'g' in text:
                    self.current_session['learned_facts'][compound].append({
                        'context': 'dosage discussed',
                        'timestamp': datetime.now().isoformat()
                    })
    
    def get_relevant_context(self, query: str, max_interactions: int = 5) -> str:
        """Get relevant context for current query."""
        context_parts = []
        
        # 1. Recent conversation context
        recent = self.current_session['interactions'][-max_interactions:]
        if recent:
            recent_text = "Recent conversation:\n"
            for interaction in recent:
                recent_text += f"User: {interaction['user']}\n"
                recent_text += f"Assistant: {interaction['assistant'][:200]}...\n\n"
            context_parts.append(recent_text)
        
        # 2. Relevant learned facts
        query_lower = query.lower()
        relevant_facts = []
        
        for fact_key, fact_value in self.current_session['learned_facts'].items():
            if fact_key in query_lower:
                relevant_facts.append(f"Previous discussion about {fact_key}: {fact_value}")
        
        if relevant_facts:
            context_parts.append("Relevant information from our conversation:\n" + "\n".join(relevant_facts))
        
        # 3. User preferences
        if self.current_session['user_preferences']:
            prefs = "User preferences:\n"
            for key, value in self.current_session['user_preferences'].items():
                prefs += f"- {key}: {value}\n"
            context_parts.append(prefs)
        
        return "\n\n".join(context_parts)
    
    def should_ask_clarification(self, user_message: str) -> Tuple[bool, Optional[str]]:
        """Determine if clarification is needed."""
        # Check for vague supplement requests
        vague_indicators = [
            'something for',
            'supplement for',
            'help with',
            'improve my',
            'what about'
        ]
        
        message_lower = user_message.lower()
        
        # Specific checks for pharmaceutical development
        if any(indicator in message_lower for indicator in vague_indicators):
            if 'muscle' in message_lower and 'dose' not in message_lower:
                return True, "What's your current training experience and are there any specific muscle groups or performance metrics you want to target?"
            
            if 'energy' in message_lower and 'when' not in message_lower:
                return True, "When do you need the energy boost - pre-workout, general daily energy, or specific times? This helps me recommend the right compound and timing."
            
            if 'develop' in message_lower or 'create' in message_lower:
                if 'for' in message_lower and len(message_lower.split()) < 10:
                    return True, "Could you provide more details about the target population, intended use case, and any specific requirements (e.g., natural ingredients, specific price point)?"
        
        return False, None
    
    def update_user_preferences(self, key: str, value: str):
        """Update learned user preferences."""
        self.current_session['user_preferences'][key] = value
        self.persistent_memory['preferences'][key] = value
        self._save_persistent_memory()
    
    def _save_session(self):
        """Save current session to disk."""
        session_file = self.memory_dir / f"session_{self.current_session['id']}.json"
        with open(session_file, 'w') as f:
            json.dump(self.current_session, f, indent=2)
    
    def _save_persistent_memory(self):
        """Save persistent memory."""
        memory_file = self.memory_dir / "persistent_memory.json"
        with open(memory_file, 'w') as f:
            json.dump(self.persistent_memory, f, indent=2)
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.current_session['interactions']:
            return "No conversation yet."
        
        summary = f"Conversation started: {self.current_session['start_time']}\n"
        summary += f"Total interactions: {len(self.current_session['interactions'])}\n"
        
        if self.current_session['learned_facts']:
            summary += "\nTopics discussed:\n"
            for topic in self.current_session['learned_facts'].keys():
                summary += f"- {topic}\n"
        
        return summary

# Singleton instance
_memory_instance = None

def get_conversation_memory() -> ConversationMemory:
    """Get or create the global conversation memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance 