"""
Integration module to connect enhanced pharmaceutical RAG with the GUI
Provides seamless switching between standard and pharma-enhanced responses
"""
from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path

from .pharma_rag_enhanced import get_pharma_rag
from .knowledge_gap_analyzer import KnowledgeGapAnalyzer
from .enhanced_rag_chat import EnhancedRAGChat, answer_medical_query
from .rag_chat import answer as standard_answer

logger = logging.getLogger(__name__)


async def enhanced_ai_handler(message: str, history: List[List[str]]) -> str:
    """
    Enhanced AI handler with medical research capabilities.
    Fixes output format issues and conducts autonomous research.
    """
    try:
        # Use the enhanced medical RAG chat
        response = await answer_medical_query(message, history)
        return response
        
    except Exception as e:
        # Fallback to basic pharma RAG if enhanced fails
        try:
            pharma_rag = get_pharma_rag()
            knowledge_items = pharma_rag.retrieve(message, top_k=5)
            
            if knowledge_items:
                # Format response with answer first
                response = f"Based on my pharmaceutical knowledge:\n\n"
                
                # Synthesize answer from knowledge
                answer = _synthesize_answer(message, knowledge_items)
                response += answer
                
                # Add sources at the end
                response += "\n\n**Sources:**\n"
                for i, item in enumerate(knowledge_items[:3], 1):
                    response += f"{i}. {item['metadata'].get('source', 'Knowledge base')}\n"
                    
                return response
            else:
                return "I don't have specific information about that in my knowledge base. Please try rephrasing your question."
                
        except Exception as fallback_error:
            return f"I encountered an error: {str(e)}. Please try again or rephrase your question."


def _synthesize_answer(question: str, knowledge_items: List[Dict]) -> str:
    """Synthesize an answer from knowledge items."""
    # Simple synthesis - in production use LLM
    if not knowledge_items:
        return "No relevant information found."
        
    # Extract key information
    key_points = []
    for item in knowledge_items[:3]:
        content = item.get('content', '')
        if len(content) > 100:
            key_points.append(content[:200] + "...")
        else:
            key_points.append(content)
            
    # Format as coherent answer
    answer = "Here's what I found:\n\n"
    for i, point in enumerate(key_points, 1):
        answer += f"{i}. {point}\n\n"
        
    return answer


def _is_pharmaceutical_question(message: str) -> bool:
    """Detect if a question is pharmaceutical/biomedical in nature."""
    
    pharma_keywords = [
        # Drugs
        'drug', 'medication', 'medicine', 'pharmaceutical', 'compound',
        'metformin', 'aspirin', 'insulin', 'statin', 'antibiotic',
        
        # Medical terms
        'disease', 'treatment', 'therapy', 'clinical', 'patient',
        'diagnosis', 'symptom', 'side effect', 'adverse',
        
        # Molecular/biological
        'protein', 'enzyme', 'receptor', 'pathway', 'mechanism',
        'molecular', 'binding', 'target', 'inhibitor', 'agonist',
        
        # Pharmacology
        'pharmacokinetics', 'pharmacodynamics', 'metabolism',
        'absorption', 'distribution', 'elimination', 'bioavailability',
        
        # Research
        'trial', 'study', 'research', 'efficacy', 'safety'
    ]
    
    message_lower = message.lower()
    
    # Count matching keywords
    matches = sum(1 for keyword in pharma_keywords if keyword in message_lower)
    
    # Consider it pharmaceutical if 2+ keywords match or certain key terms are present
    if matches >= 2:
        return True
        
    # Check for specific strong indicators
    strong_indicators = ['mechanism of action', 'clinical trial', 'drug interaction', 
                        'side effect', 'treatment for']
    
    return any(indicator in message_lower for indicator in strong_indicators)


def create_pharma_aware_chat_interface():
    """Create a chat interface that's aware of pharmaceutical research needs."""
    
    import gradio as gr
    
    def sync_handler(message: str, history: List[List[str]]) -> str:
        """Synchronous wrapper for async handler."""
        return asyncio.run(enhanced_ai_handler(message, history))
    
    examples = [
        "What is the mechanism of action of metformin?",
        "Compare the efficacy of GLP-1 agonists vs SGLT2 inhibitors for diabetes",
        "What are the drug-drug interactions of warfarin?",
        "Explain the mTOR pathway and its role in aging",
        "What are the latest clinical trials for Alzheimer's treatment?",
        "How does creatine supplementation affect muscle metabolism?",
        "What are the molecular targets of statins?",
        "Analyze the pharmacokinetics of aspirin"
    ]
    
    interface = gr.ChatInterface(
        fn=sync_handler,
        title="🧬 Pharmaceutical Research Assistant",
        description="""
        I'm your specialized pharmaceutical research assistant. I can help with:
        • Drug mechanisms and interactions
        • Clinical trial analysis
        • Molecular pathways
        • Treatment comparisons
        • Safety profiles
        
        I'll tell you when I need more data to give you a complete answer!
        """,
        examples=examples,
        theme=gr.themes.Soft()
    )
    
    return interface


def get_knowledge_coverage_report(query: str) -> Dict[str, Any]:
    """
    Get a detailed report on knowledge coverage for a query.
    Useful for understanding what the system knows and doesn't know.
    """
    
    try:
        # Initialize components
        pharma_rag = get_pharma_rag()
        gap_analyzer = KnowledgeGapAnalyzer()
        
        # Get available documents
        docs = pharma_rag.retrieve(query)
        doc_contents = [d if isinstance(d, str) else d.get('content', '') for d in docs]
        
        # Analyze coverage
        coverage_analysis = gap_analyzer.analyze_query_coverage(query, doc_contents)
        
        # Generate detailed report
        report = {
            'query': query,
            'knowledge_coverage': {
                'score': coverage_analysis['coverage_score'],
                'confidence': coverage_analysis['confidence'],
                'covered_topics': coverage_analysis['covered_aspects'],
                'missing_topics': coverage_analysis['missing_aspects']
            },
            'knowledge_gaps': coverage_analysis['specific_gaps'],
            'recommended_sources': coverage_analysis['suggested_sources'],
            'query_complexity': coverage_analysis.get('query_type', 'unknown'),
            'improvement_suggestions': [
                f"Add data from {source['source']}" 
                for source in coverage_analysis['suggested_sources'][:3]
            ]
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Knowledge coverage report error: {str(e)}")
        return {
            'error': str(e),
            'query': query,
            'knowledge_coverage': {'score': 0, 'confidence': 0}
        } 