"""
Enhanced RAG System for Pharmaceutical Research
Specialized for drug discovery, molecular analysis, and clinical research
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
from .rag_chat import RAGChat, get_chat
from .embeddings import Embedder

logger = logging.getLogger(__name__)


class PharmaKnowledgeCategories:
    """Categories of pharmaceutical knowledge."""
    
    DRUG_INTERACTIONS = "drug_interactions"
    MOLECULAR_PROPERTIES = "molecular_properties"
    CLINICAL_TRIALS = "clinical_trials"
    DISEASE_PATHWAYS = "disease_pathways"
    PROTEIN_TARGETS = "protein_targets"
    PHARMACOKINETICS = "pharmacokinetics"
    ADVERSE_EVENTS = "adverse_events"
    REGULATORY_INFO = "regulatory_info"


class PharmaRAGEnhanced:
    """Enhanced RAG system specialized for pharmaceutical research.
    
    Uses composition instead of inheritance to avoid duplicate instances.
    """
    
    def __init__(self):
        # Use the global RAG instance instead of creating a new one
        self.rag_chat = get_chat()
        self.knowledge_categories = {}
        self.research_context = {}
        self.missing_knowledge = []
        
        # Load pharma-specific templates
        self._load_pharma_templates()
        
    # Delegate core RAG methods to the composed instance
    def retrieve(self, query: str):
        """Retrieve documents using the underlying RAG system."""
        return self.rag_chat.retrieve(query)
    
    async def generate(self, prompt: str):
        """Generate text using the underlying RAG system."""
        return await self.rag_chat.generate(prompt)
    
    async def answer(self, question: str):
        """Answer questions using the underlying RAG system."""
        return await self.rag_chat.answer(question)
    
    def is_ready(self):
        """Check if the underlying RAG system is ready."""
        return self.rag_chat.is_ready()
        
    def _load_pharma_templates(self):
        """Load pharmaceutical research templates."""
        self.templates = {
            "drug_analysis": """
As a pharmaceutical researcher, analyze the following drug information:

Drug: {drug_name}
Context: {context}

Please provide:
1. Mechanism of action
2. Primary targets
3. Known interactions
4. Clinical applications
5. Safety profile

Base your analysis on the following sources:
{sources}
""",
            "molecular_analysis": """
Analyze the molecular properties of {compound}:

Available data:
{context}

Required analysis:
1. Chemical structure insights
2. ADMET properties
3. Potential biological activity
4. Structure-activity relationships
5. Synthesis considerations

References:
{sources}
""",
            "clinical_research": """
Evaluate the clinical research for {topic}:

Research context:
{context}

Analysis required:
1. Current clinical trial status
2. Efficacy data
3. Safety profile
4. Patient populations
5. Regulatory considerations

Based on:
{sources}
""",
            "knowledge_gap": """
Based on the query about {topic}, I've identified the following knowledge gaps:

{gaps}

To provide a comprehensive answer, I would need:
{needed_data}

Currently available information:
{available_info}
"""
        }
        
    async def pharma_answer(self, question: str, research_type: str = None) -> Dict[str, Any]:
        """
        Answer pharmaceutical research questions with enhanced context.
        
        Returns dict with:
        - answer: The main response
        - confidence: Confidence score (0-1)
        - sources: List of sources used
        - knowledge_gaps: Identified missing information
        - suggestions: Next research steps
        """
        
        # Analyze question type
        question_analysis = self._analyze_pharma_question(question)
        research_type = research_type or question_analysis['type']
        
        # Retrieve specialized documents
        docs = await self._retrieve_pharma_docs(question, research_type)
        
        # Check knowledge coverage
        coverage = self._assess_knowledge_coverage(question, docs)
        
        # Generate enhanced response
        if coverage['score'] < 0.5:
            # Insufficient knowledge - provide gap analysis
            response = self._generate_knowledge_gap_response(question, coverage)
        else:
            # Generate comprehensive pharma response
            response = await self._generate_pharma_response(
                question, docs, research_type, question_analysis
            )
            
        # Add metadata
        response['confidence'] = coverage['score']
        response['knowledge_gaps'] = coverage['gaps']
        response['suggestions'] = self._generate_research_suggestions(
            question, coverage, response
        )
        
        return response
        
    def _analyze_pharma_question(self, question: str) -> Dict[str, Any]:
        """Analyze the pharmaceutical research question."""
        question_lower = question.lower()
        
        # Detect question type
        if any(word in question_lower for word in ['drug', 'medication', 'compound']):
            q_type = PharmaKnowledgeCategories.DRUG_INTERACTIONS
        elif any(word in question_lower for word in ['molecular', 'structure', 'chemical']):
            q_type = PharmaKnowledgeCategories.MOLECULAR_PROPERTIES
        elif any(word in question_lower for word in ['clinical', 'trial', 'study']):
            q_type = PharmaKnowledgeCategories.CLINICAL_TRIALS
        elif any(word in question_lower for word in ['disease', 'pathway', 'mechanism']):
            q_type = PharmaKnowledgeCategories.DISEASE_PATHWAYS
        elif any(word in question_lower for word in ['protein', 'target', 'receptor']):
            q_type = PharmaKnowledgeCategories.PROTEIN_TARGETS
        else:
            q_type = "general"
            
        # Extract key entities
        entities = self._extract_pharma_entities(question)
        
        return {
            'type': q_type,
            'entities': entities,
            'complexity': self._assess_question_complexity(question)
        }
        
    def _extract_pharma_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract pharmaceutical entities from text."""
        # Simplified entity extraction
        # In production, use BioBERT or similar
        entities = {
            'drugs': [],
            'diseases': [],
            'proteins': [],
            'pathways': []
        }
        
        # Common drug names
        drug_keywords = ['metformin', 'aspirin', 'ibuprofen', 'insulin', 'statins', 
                        'warfarin', 'creatine', 'dopamine', 'serotonin']
        
        # Common disease terms
        disease_keywords = ['diabetes', 'cancer', 'hypertension', 'alzheimer', 
                           'parkinson', 'covid', 'inflammation']
        
        # Check for entities
        text_lower = text.lower()
        for drug in drug_keywords:
            if drug in text_lower:
                entities['drugs'].append(drug)
                
        for disease in disease_keywords:
            if disease in text_lower:
                entities['diseases'].append(disease)
                
        return entities
        
    async def _retrieve_pharma_docs(self, question: str, research_type: str) -> List[Dict]:
        """Retrieve documents with pharmaceutical context."""
        # Standard retrieval
        base_docs = self.retrieve(question)
        
        # Enhance with pharma metadata
        enhanced_docs = []
        for doc in base_docs:
            enhanced = {
                'content': doc,
                'category': self._categorize_document(doc),
                'reliability': self._assess_source_reliability(doc),
                'recency': self._extract_publication_date(doc)
            }
            enhanced_docs.append(enhanced)
            
        # Sort by relevance and reliability
        enhanced_docs.sort(
            key=lambda x: (x['reliability'], -len(x['content'])), 
            reverse=True
        )
        
        return enhanced_docs
        
    def _categorize_document(self, doc: str) -> str:
        """Categorize document by pharmaceutical domain."""
        doc_lower = doc.lower()
        
        if 'clinical trial' in doc_lower:
            return PharmaKnowledgeCategories.CLINICAL_TRIALS
        elif 'adverse' in doc_lower or 'side effect' in doc_lower:
            return PharmaKnowledgeCategories.ADVERSE_EVENTS
        elif 'mechanism' in doc_lower or 'pathway' in doc_lower:
            return PharmaKnowledgeCategories.DISEASE_PATHWAYS
        else:
            return "general"
            
    def _assess_source_reliability(self, doc: str) -> float:
        """Assess reliability of the source (0-1)."""
        # Look for indicators of reliability
        reliability_indicators = {
            'peer-reviewed': 0.9,
            'clinical trial': 0.85,
            'fda': 0.95,
            'pubmed': 0.8,
            'review': 0.7,
            'case report': 0.6
        }
        
        score = 0.5  # Default
        doc_lower = doc.lower()
        for indicator, value in reliability_indicators.items():
            if indicator in doc_lower:
                score = max(score, value)
                
        return score
        
    def _extract_publication_date(self, doc: str) -> Optional[str]:
        """Extract publication date from document."""
        # Simplified - look for year patterns
        import re
        year_pattern = r'20[0-2][0-9]'
        matches = re.findall(year_pattern, doc)
        return matches[0] if matches else None
        
    def _assess_knowledge_coverage(self, question: str, docs: List[Dict]) -> Dict[str, Any]:
        """Assess how well the available knowledge covers the question."""
        coverage = {
            'score': 0.0,
            'gaps': [],
            'covered_aspects': [],
            'missing_aspects': []
        }
        
        # Analyze what the question needs
        needed_aspects = self._identify_needed_aspects(question)
        
        # Check what we have
        for aspect in needed_aspects:
            if self._is_aspect_covered(aspect, docs):
                coverage['covered_aspects'].append(aspect)
            else:
                coverage['missing_aspects'].append(aspect)
                coverage['gaps'].append(f"Missing information about: {aspect}")
                
        # Calculate coverage score
        if needed_aspects:
            coverage['score'] = len(coverage['covered_aspects']) / len(needed_aspects)
        else:
            coverage['score'] = 0.5  # Default for unclear questions
            
        return coverage
        
    def _identify_needed_aspects(self, question: str) -> List[str]:
        """Identify what aspects of information are needed."""
        aspects = []
        question_lower = question.lower()
        
        # Common pharmaceutical research aspects
        aspect_keywords = {
            'mechanism': ['mechanism', 'how does', 'work'],
            'safety': ['safe', 'side effect', 'adverse', 'risk'],
            'efficacy': ['effective', 'work', 'benefit', 'outcome'],
            'dosage': ['dose', 'dosage', 'how much', 'concentration'],
            'interactions': ['interact', 'combination', 'contraindication'],
            'clinical_data': ['clinical', 'trial', 'study', 'evidence'],
            'molecular': ['structure', 'molecular', 'chemical']
        }
        
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                aspects.append(aspect)
                
        return aspects or ['general_information']
        
    def _is_aspect_covered(self, aspect: str, docs: List[Dict]) -> bool:
        """Check if an aspect is covered in the documents."""
        aspect_patterns = {
            'mechanism': ['mechanism', 'pathway', 'target', 'bind'],
            'safety': ['adverse', 'side effect', 'toxicity', 'safe'],
            'efficacy': ['effective', 'efficacy', 'outcome', 'response rate'],
            'dosage': ['mg', 'dose', 'concentration', 'administration'],
            'interactions': ['interaction', 'contraindicated', 'combination'],
            'clinical_data': ['trial', 'patient', 'study', 'clinical'],
            'molecular': ['structure', 'formula', 'molecular weight']
        }
        
        patterns = aspect_patterns.get(aspect, [aspect])
        
        for doc in docs:
            doc_content = doc.get('content', '').lower()
            if any(pattern in doc_content for pattern in patterns):
                return True
                
        return False
        
    def _generate_knowledge_gap_response(self, question: str, coverage: Dict) -> Dict[str, Any]:
        """Generate response highlighting knowledge gaps."""
        gaps_text = "\n".join(f"• {gap}" for gap in coverage['gaps'])
        needed_text = "\n".join(f"• {aspect}" for aspect in coverage['missing_aspects'])
        available_text = "\n".join(f"• {aspect}" for aspect in coverage['covered_aspects'])
        
        response_text = self.templates['knowledge_gap'].format(
            topic=question,
            gaps=gaps_text or "No specific gaps identified",
            needed_data=needed_text or "Additional clinical or molecular data",
            available_info=available_text or "Limited information available"
        )
        
        return {
            'answer': response_text,
            'sources': [],
            'needs_more_data': True
        }
        
    async def _generate_pharma_response(
        self, 
        question: str, 
        docs: List[Dict], 
        research_type: str,
        question_analysis: Dict
    ) -> Dict[str, Any]:
        """Generate comprehensive pharmaceutical research response."""
        
        # Select appropriate template
        if research_type == PharmaKnowledgeCategories.DRUG_INTERACTIONS:
            template = self.templates['drug_analysis']
        elif research_type == PharmaKnowledgeCategories.MOLECULAR_PROPERTIES:
            template = self.templates['molecular_analysis']
        elif research_type == PharmaKnowledgeCategories.CLINICAL_TRIALS:
            template = self.templates['clinical_research']
        else:
            # Use standard RAG
            standard_response = await self.answer(question)
            return {
                'answer': standard_response,
                'sources': self._extract_sources(docs),
                'needs_more_data': False
            }
            
        # Build context from documents
        context = self._build_pharma_context(docs, research_type)
        sources = self._format_sources(docs)
        
        # Get entity name for template
        entity_name = 'the compound'
        if question_analysis['entities'].get('drugs'):
            entity_name = question_analysis['entities']['drugs'][0]
        
        # Fill template
        prompt = template.format(
            drug_name=entity_name,
            compound=entity_name,
            topic=question,
            context=context,
            sources=sources
        )
        
        # Generate response
        response = await self.generate(prompt)
        
        return {
            'answer': response,
            'sources': self._extract_sources(docs),
            'needs_more_data': False
        }
        
    def _build_pharma_context(self, docs: List[Dict], research_type: str) -> str:
        """Build specialized context for pharmaceutical queries."""
        # Group documents by category
        categorized = {}
        for doc in docs:
            category = doc.get('category', 'general')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(doc['content'])
            
        # Build structured context
        context_parts = []
        for category, contents in categorized.items():
            context_parts.append(f"\n{category.upper()}:")
            for content in contents[:2]:  # Limit per category
                context_parts.append(f"- {content[:200]}...")
                
        return "\n".join(context_parts)
        
    def _format_sources(self, docs: List[Dict]) -> str:
        """Format sources with reliability indicators."""
        sources = []
        for i, doc in enumerate(docs[:5], 1):
            reliability = doc.get('reliability', 0.5)
            date = doc.get('recency', 'Unknown date')
            sources.append(
                f"[{i}] (Reliability: {reliability:.1%}, Date: {date}) "
                f"{doc['content'][:100]}..."
            )
        return "\n".join(sources)
        
    def _extract_sources(self, docs: List[Dict]) -> List[str]:
        """Extract clean source list."""
        return [doc.get('content', '')[:200] + "..." for doc in docs[:5]]
        
    def _generate_research_suggestions(
        self, 
        question: str, 
        coverage: Dict,
        response: Dict
    ) -> List[str]:
        """Generate next steps for research."""
        suggestions = []
        
        if coverage['score'] < 0.7:
            suggestions.append("Consider searching for recent clinical trial data")
            suggestions.append("Look for systematic reviews or meta-analyses")
            
        if coverage['missing_aspects']:
            for aspect in coverage['missing_aspects']:
                if aspect == 'safety':
                    suggestions.append("Search FDA adverse event databases")
                elif aspect == 'mechanism':
                    suggestions.append("Review molecular pathway databases")
                elif aspect == 'clinical_data':
                    suggestions.append("Check ClinicalTrials.gov for ongoing studies")
                    
        if not suggestions:
            suggestions.append("Current information appears comprehensive")
            
        return suggestions
        
    def _assess_question_complexity(self, question: str) -> str:
        """Assess complexity of the research question."""
        word_count = len(question.split())
        technical_terms = sum(1 for word in ['pharmacokinetics', 'bioavailability', 
                                             'cytochrome', 'receptor', 'inhibitor', 
                                             'agonist', 'metabolite']
                             if word in question.lower())
        
        if word_count > 30 or technical_terms > 2:
            return 'complex'
        elif word_count > 15 or technical_terms > 0:
            return 'moderate'
        else:
            return 'simple'


# Global instance
_pharma_rag: Optional[PharmaRAGEnhanced] = None


def get_pharma_rag() -> PharmaRAGEnhanced:
    """Get or create the global pharma RAG instance."""
    global _pharma_rag
    if _pharma_rag is None:
        _pharma_rag = PharmaRAGEnhanced()
    return _pharma_rag 