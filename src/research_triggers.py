"""
Research Triggers - Automatic research detection and initiation
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional
import asyncio
from datetime import datetime

from .experimental_research_engine import ExperimentalResearchEngine, integrate_with_graph

# Research trigger phrases
RESEARCH_TRIGGERS = [
    "research",
    "investigate", 
    "study",
    "analyze",
    "explore",
    "find out",
    "discover",
    "test hypothesis",
    "experiment with",
    "what if",
    "how can we",
    "is it possible",
    "deep dive",
    "comprehensive analysis"
]

# Pre-made research prompts
RESEARCH_PROMPTS = {
    "muscle_growth": "Research the most effective methods to maximize muscle protein synthesis and hypertrophy",
    "recovery": "Investigate optimal recovery strategies between training sessions",
    "supplements": "Research which supplements have the strongest evidence for performance enhancement",
    "training": "Analyze the most effective training protocols for strength and muscle gains",
    "nutrition": "Research optimal nutrition timing and macronutrient ratios for athletes",
    "sleep": "Investigate how sleep quality affects muscle growth and athletic performance",
    "creatine": "Research ways to enhance creatine absorption and effectiveness",
    "fatigue": "Study mechanisms of fatigue and how to delay its onset during training",
    "metabolism": "Investigate methods to optimize metabolic efficiency for performance",
    "longevity": "Research interventions that improve both performance and healthspan"
}


class ResearchTriggerDetector:
    """Detects when to trigger autonomous research."""
    
    def __init__(self):
        self.active_research = None
        self.research_log = []
        
    def should_trigger_research(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Check if message should trigger research.
        Returns (should_trigger, research_type)
        """
        message_lower = message.lower()
        
        # Check for explicit research triggers
        for trigger in RESEARCH_TRIGGERS:
            if trigger in message_lower:
                # Extract what to research
                research_topic = self._extract_research_topic(message, trigger)
                return True, research_topic
                
        # Check for question patterns that benefit from research
        if self._is_complex_question(message):
            return True, message
            
        return False, None
        
    def _extract_research_topic(self, message: str, trigger: str) -> str:
        """Extract the research topic from the message."""
        # Find text after the trigger word
        pattern = rf"{trigger}\s+(.+?)(?:\.|$)"
        match = re.search(pattern, message.lower())
        
        if match:
            return match.group(1).strip()
        
        # Fallback to full message
        return message
        
    def _is_complex_question(self, message: str) -> bool:
        """Check if question is complex enough to warrant research."""
        complex_indicators = [
            "how to maximize",
            "what is the best",
            "most effective",
            "optimize",
            "enhance",
            "improve",
            "compare",
            "vs",
            "better than",
            "mechanisms",
            "scientific evidence"
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in complex_indicators)
        
    async def trigger_research(self, topic: str, max_iterations: int = 5) -> Dict:
        """
        Trigger autonomous research on a topic.
        
        Args:
            topic: The research topic
            max_iterations: Number of research iterations
                - 5: Quick research (5-10 minutes)
                - 10: Standard research (10-20 minutes)  
                - 20: Deep research (20-40 minutes)
                - 30: Comprehensive research (30-60 minutes)
        """
        # Determine research depth based on topic
        if any(word in topic.lower() for word in ['comprehensive', 'deep', 'thorough', 'extensive']):
            max_iterations = max(max_iterations, 20)  # At least 20 iterations for deep research
            self.log_research_event(f"🔬 Starting DEEP research (est. 20-40 min): {topic}")
        elif any(word in topic.lower() for word in ['quick', 'brief', 'summary']):
            max_iterations = min(max_iterations, 5)  # Quick research
            self.log_research_event(f"⚡ Starting quick research (est. 5-10 min): {topic}")
        else:
            # Standard research
            max_iterations = max(max_iterations, 10)
            self.log_research_event(f"🚀 Starting standard research (est. 10-20 min): {topic}")
        
        # Create research engine
        engine = ExperimentalResearchEngine()
        engine = integrate_with_graph(engine)
        
        # Store as active research
        self.active_research = {
            'topic': topic,
            'start_time': datetime.now(),
            'engine': engine,
            'depth': 'deep' if max_iterations >= 20 else 'standard' if max_iterations >= 10 else 'quick'
        }
        
        # Run research
        try:
            findings = await engine.research_loop(topic, max_iterations)
            summary = engine.get_summary()
            
            # Log completion
            duration = (datetime.now() - self.active_research['start_time']).total_seconds() / 60
            self.log_research_event(
                f"✅ Research complete: {summary['total_innovations']} innovations generated in {duration:.1f} minutes"
            )
            
            return {
                'success': True,
                'findings': findings,
                'summary': summary,
                'log_file': str(engine.log_file),
                'duration_minutes': duration
            }
            
        except Exception as e:
            self.log_research_event(f"❌ Research error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.active_research = None
            
    def log_research_event(self, message: str):
        """Log a research event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        self.research_log.append(event)
        
        # Keep only last 100 events
        if len(self.research_log) > 100:
            self.research_log = self.research_log[-100:]
            
    def get_research_log(self) -> List[Dict]:
        """Get the research event log."""
        return self.research_log
        
    def is_research_active(self) -> bool:
        """Check if research is currently running."""
        return self.active_research is not None
        
    def get_active_research(self) -> Optional[Dict]:
        """Get info about active research."""
        return self.active_research


# Global instance
research_detector = ResearchTriggerDetector()


def create_research_prompt(topic_key: str) -> str:
    """Create a research prompt from pre-made templates."""
    if topic_key in RESEARCH_PROMPTS:
        return RESEARCH_PROMPTS[topic_key]
    else:
        # Generate generic research prompt
        return f"Conduct comprehensive research on: {topic_key}"


def format_research_response(research_result: Dict) -> str:
    """Format research results for display."""
    if not research_result['success']:
        return f"❌ Research failed: {research_result.get('error', 'Unknown error')}"
        
    summary = research_result['summary']
    duration = research_result.get('duration_minutes', 0)
    
    response = f"""
# 🔬 Research Complete!

**Topic:** {research_result.get('topic', 'Unknown')}
**Duration:** {duration:.1f} minutes
**Iterations:** {summary['iterations']}
**Innovations Generated:** {summary['total_innovations']}

## 💡 Top Innovations:
"""
    
    for i, innovation in enumerate(summary['top_innovations'][:10], 1):  # Show more innovations
        response += f"{i}. {innovation}\n"
        
    response += f"\n📄 **Full research log saved to:** {research_result.get('log_file', 'research_log.md')}"
    
    # Add research quality indicator
    if duration >= 20:
        response += "\n\n🏆 **Deep Research:** This was a comprehensive investigation with thorough analysis."
    elif duration >= 10:
        response += "\n\n📊 **Standard Research:** Good coverage of the topic with solid findings."
    else:
        response += "\n\n⚡ **Quick Research:** Fast exploration of key concepts."
    
    return response 