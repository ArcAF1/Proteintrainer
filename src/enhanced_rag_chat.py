"""
Enhanced RAG Chat with Medical Research Integration
Fixes output format issues and integrates autonomous research
"""
from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional
import logging

from .medical_research_agent import MedicalResearchAgent
from .rag_chat import RAGChat

logger = logging.getLogger(__name__)


class EnhancedRAGChat(RAGChat):
    """
    Enhanced RAG chat that uses the medical research agent
    and enforces proper output formatting.
    """
    
    def __init__(self):
        super().__init__()
        self.medical_agent = MedicalResearchAgent()
        
        # Override the default prompt to fix output issues
        self.system_prompt = """You are a medical AI assistant. IMPORTANT RULES:

1. ALWAYS provide a direct answer FIRST
2. NEVER start with sources or citations
3. List sources ONLY at the end of your response
4. Use clear, structured formatting

When answering:
- Start with a comprehensive answer
- Explain mechanisms and effects
- Provide practical recommendations
- End with sources (if any)

Remember: Users want answers, not just references."""
        
    async def generate(self, prompt: str, context: str = "", history: List[List[str]] = None) -> str:
        """
        Generate response with medical research capabilities.
        Automatically conducts research when needed.
        """
        # Check if this is a medical/research query
        if self._is_medical_query(prompt):
            # Use medical research agent
            try:
                result = await self.medical_agent.answer_query(prompt)
                
                # Format the response
                response = self._format_medical_response(result)
                
                # Log research activity
                if result.get('research_conducted'):
                    logger.info(f"Conducted autonomous research for: {prompt}")
                    logger.info(f"Filled knowledge gaps: {result.get('knowledge_gaps_filled', [])}")
                    
                return response
                
            except Exception as e:
                logger.error(f"Medical research error: {str(e)}")
                # Fallback to standard RAG
                
        # Use standard RAG for non-medical queries or on error
        return await super().generate(prompt, context, history)
        
    def _is_medical_query(self, prompt: str) -> bool:
        """Determine if query requires medical research."""
        medical_indicators = [
            'drug', 'medication', 'treatment', 'disease', 'symptom',
            'side effect', 'mechanism', 'clinical', 'medical', 'health',
            'alternative', 'supplement', 'therapy', 'condition',
            'creatine', 'protein', 'vitamin', 'mineral'
        ]
        
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in medical_indicators)
        
    def _format_medical_response(self, result: Dict[str, Any]) -> str:
        """Format medical research results into user-friendly response."""
        response_parts = []
        
        # Main answer
        response_parts.append(result['answer'])
        
        # Add confidence indicator
        confidence = result.get('confidence', 0.5)
        if confidence < 0.5:
            response_parts.append("\n\n⚠️ **Note:** This answer is based on limited evidence. Consider consulting healthcare professionals.")
        elif confidence > 0.8:
            response_parts.append("\n\n✅ **Confidence:** High - based on multiple quality sources")
            
        # Add research indicator
        if result.get('research_conducted'):
            gaps = result.get('knowledge_gaps_filled', [])
            response_parts.append(f"\n\n🔬 **Research Update:** I conducted new research to answer your question, filling knowledge gaps about: {', '.join(gaps)}")
            
        # Add sources
        if result.get('sources'):
            response_parts.append("\n\n**Sources:**")
            for i, source in enumerate(result['sources'][:5], 1):
                response_parts.append(f"{i}. {source}")
                
        return "\n".join(response_parts)


async def answer_medical_query(query: str, history: List[List[str]] = None) -> str:
    """
    Main entry point for medical queries.
    Uses enhanced RAG with medical research capabilities.
    """
    chat = EnhancedRAGChat()
    return await chat.generate(query, history=history)


# Example usage functions

async def find_creatine_alternatives():
    """Example: Find alternatives to creatine."""
    query = "What gives the same effect as creatine but is legal and better?"
    
    agent = MedicalResearchAgent()
    result = await agent.find_alternatives(
        "creatine",
        ["legal", "better", "same effect"]
    )
    
    print("=== CREATINE ALTERNATIVES ===")
    print(f"\nFull Analysis:\n{result['full_analysis']}")
    print(f"\nConfidence: {result['confidence']:.1%}")
    print(f"\nResearch Conducted: {result['research_conducted']}")
    
    if result['alternatives']:
        print("\nSpecific Alternatives Found:")
        for alt in result['alternatives']:
            print(f"- {alt['name']}")
            
    return result


async def research_muscle_recovery():
    """Example: Research muscle recovery optimization."""
    query = "How can we optimize muscle recovery after high-intensity training?"
    
    agent = MedicalResearchAgent()
    result = await agent.answer_query(query)
    
    print("=== MUSCLE RECOVERY RESEARCH ===")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']:.1%}")
    
    if result.get('knowledge_gaps_filled'):
        print(f"\nKnowledge gaps filled: {', '.join(result['knowledge_gaps_filled'])}")
        
    return result


# Prompt templates for fixing Mistral output

MISTRAL_SYSTEM_PROMPT = """You are a helpful medical AI assistant with access to a comprehensive knowledge base.

CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
1. ALWAYS start with a direct answer to the question
2. Provide comprehensive information in clear paragraphs
3. Use examples and explanations as needed
4. ONLY list sources at the very END of your response
5. NEVER start your response with sources, citations, or references

RESPONSE STRUCTURE:
[Direct answer and explanation - multiple paragraphs]
[Additional details if relevant]
[Practical recommendations if applicable]

Sources: [List any sources here at the end]

Remember: The user wants an ANSWER first, not a list of papers."""


MISTRAL_ANSWER_TEMPLATE = """Based on my knowledge base and research, here's what I found about {topic}:

[MAIN ANSWER]
{answer_content}

[MECHANISMS/DETAILS]
{mechanism_details}

[PRACTICAL APPLICATIONS]
{practical_info}

[SOURCES]
{source_list}""" 