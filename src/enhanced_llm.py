"""
Enhanced LLM with Biomedical Focus
Integrates system prompts, conversation memory, and clarification questions
"""
import logging
from typing import Optional, Dict, Any
import llama_cpp

from .config import settings
from .biomedical_system_prompt import get_enhanced_prompt, CLARIFICATION_PROMPT
from .conversation_memory import get_conversation_memory

logger = logging.getLogger(__name__)

class EnhancedBiomedicalLLM:
    """Enhanced LLM with biomedical specialization and memory."""
    
    def __init__(self):
        self.llm = self._load_model()
        self.memory = get_conversation_memory()
        
    def _load_model(self):
        """Load the LLM with optimized settings."""
        model_path = settings.model_dir / settings.llm_model
        
        # Check active configuration for GPU settings
        config = self._load_active_config()
        
        llm = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=config.get('n_ctx', 4096),
            n_gpu_layers=config.get('n_gpu_layers', 24),
            n_threads=config.get('n_threads', 8),
            use_mlock=config.get('use_mlock', False),
            n_batch=config.get('n_batch', 512),
            verbose=False
        )
        
        logger.info(f"Enhanced LLM loaded with {config.get('n_gpu_layers', 0)} GPU layers")
        return llm
    
    def _load_active_config(self) -> Dict[str, Any]:
        """Load active configuration."""
        import json
        from pathlib import Path
        
        config_path = Path("active_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                return config_data.get('llm_config', config_data.get('llm', {}))
        
        # Default config
        return {
            'n_ctx': 4096,
            'n_gpu_layers': 24,
            'n_threads': 8,
            'use_mlock': False,
            'n_batch': 512
        }
    
    def generate(self, user_message: str, **kwargs) -> str:
        """Generate response with enhanced biomedical focus."""
        
        # Check if clarification is needed
        needs_clarification, clarification_question = self.memory.should_ask_clarification(user_message)
        if needs_clarification:
            # Ask clarifying question
            response = clarification_question
            self.memory.add_interaction(user_message, response, {"type": "clarification"})
            return response
        
        # Get relevant context from memory
        context = self.memory.get_relevant_context(user_message)
        
        # Build enhanced prompt
        prompt = get_enhanced_prompt(user_message, context)
        
        # Generate response
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 600),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.95),
            stop=["Human:", "User:", "\n\n\n"],
            echo=False
        )
        
        if isinstance(result, dict) and 'choices' in result:
            response = result['choices'][0]['text'].strip()
        else:
            response = str(result).strip()
        
        # Store interaction in memory
        self.memory.add_interaction(user_message, response)
        
        return response
    
    def explain_capabilities(self) -> str:
        """Explain what the AI can do."""
        capabilities_prompt = """Explain your capabilities in a friendly, clear way. Focus on:
1. Your expertise in pharmaceutical and supplement development
2. The extensive biomedical data you have access to
3. How you can help with specific tasks
4. Your ability to remember conversations and learn

Be specific with examples."""
        
        prompt = get_enhanced_prompt("What can you help me with?")
        
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=800,
            temperature=0.3,  # Lower temperature for consistent explanation
            echo=False
        )
        
        if isinstance(result, dict) and 'choices' in result:
            return result['choices'][0]['text'].strip()
        return str(result).strip()
    
    def develop_supplement(self, requirements: Dict[str, Any]) -> str:
        """Specialized method for supplement development."""
        # Build a detailed prompt for supplement development
        dev_prompt = f"""Develop a supplement formulation based on these requirements:

Target Goal: {requirements.get('goal', 'general health')}
Target Population: {requirements.get('population', 'general adults')}
Constraints: {requirements.get('constraints', 'none specified')}
Preferences: {requirements.get('preferences', 'none specified')}

Provide:
1. Recommended ingredients with specific doses
2. Scientific rationale for each ingredient
3. Potential synergies between ingredients
4. Safety considerations and contraindications
5. Suggested usage instructions
6. Manufacturing considerations"""

        prompt = get_enhanced_prompt(dev_prompt)
        
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.5,
            echo=False
        )
        
        if isinstance(result, dict) and 'choices' in result:
            response = result['choices'][0]['text'].strip()
        else:
            response = str(result).strip()
            
        # Store this development project in memory
        self.memory.add_interaction(
            f"Develop supplement for {requirements.get('goal')}", 
            response,
            {"type": "supplement_development", "requirements": requirements}
        )
        
        return response
    
    def analyze_compound(self, compound_name: str, analysis_type: str = "comprehensive") -> str:
        """Analyze a specific compound."""
        analysis_prompt = f"""Provide a {analysis_type} analysis of {compound_name}:

Include:
1. Mechanism of action
2. Bioavailability and pharmacokinetics
3. Typical dosing ranges
4. Clinical evidence and studies
5. Safety profile and side effects
6. Drug interactions
7. Regulatory status
8. Formulation considerations"""

        prompt = get_enhanced_prompt(analysis_prompt)
        
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=800,
            temperature=0.4,
            echo=False
        )
        
        if isinstance(result, dict) and 'choices' in result:
            return result['choices'][0]['text'].strip()
        return str(result).strip()

# Global instance
_enhanced_llm = None

def get_enhanced_llm() -> EnhancedBiomedicalLLM:
    """Get or create the enhanced LLM instance."""
    global _enhanced_llm
    if _enhanced_llm is None:
        _enhanced_llm = EnhancedBiomedicalLLM()
    return _enhanced_llm

# Convenience functions
def generate_biomedical_response(user_message: str, **kwargs) -> str:
    """Generate a biomedical-focused response."""
    llm = get_enhanced_llm()
    return llm.generate(user_message, **kwargs)

def explain_ai_capabilities() -> str:
    """Get explanation of AI capabilities."""
    llm = get_enhanced_llm()
    return llm.explain_capabilities() 