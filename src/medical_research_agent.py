"""
Autonomous Medical Research Agent
Conducts independent research using PubMed, ArXiv, and other medical databases
"""
from __future__ import annotations

import asyncio
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from xml.etree import ElementTree as ET
import arxiv

import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .autonomous_research_agent import AutonomousResearchAgent
from .research_logger import ResearchLogger
from .research_library import ResearchLibrary

logger = logging.getLogger(__name__)


class MedicalResearchAgent(AutonomousResearchAgent):
    """
    Enhanced autonomous research agent specialized for medical research.
    Integrates with PubMed, ArXiv, ClinicalTrials.gov, and other medical databases.
    """
    
    def __init__(self, research_dir: str = "medical_research"):
        super().__init__(research_dir)
        
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.research_dir / "chroma_db")
        )
        
        # Create or get collection
        self.knowledge_collection = self.chroma_client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"description": "Medical research knowledge base"}
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # API endpoints
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.clinicaltrials_base = "https://clinicaltrials.gov/api/query/study_fields"
        
        # Enhanced prompts for Mistral 7B
        self.system_prompts = self._load_optimized_prompts()
        
    def _load_optimized_prompts(self) -> Dict[str, str]:
        """Load optimized prompts for Mistral 7B to fix output format issues."""
        return {
            "answer_first": """You are a medical research AI. Follow these rules EXACTLY:

1. FIRST: Provide a direct, comprehensive answer to the question
2. THEN: List sources at the end
3. NEVER start with sources or citations
4. ALWAYS answer the question directly first

Format your response like this:
[ANSWER]
(Your comprehensive answer here - multiple paragraphs if needed)

[SOURCES]
(List sources here if any)

Remember: Answer FIRST, sources LAST.""",

            "research_mode": """You are an autonomous medical researcher. When you encounter knowledge gaps:

1. Identify what specific information is missing
2. Formulate precise research queries
3. Search relevant databases
4. Synthesize findings
5. Update your knowledge base

Always think step-by-step and document your research process.""",

            "mechanism_analysis": """Analyze medical/biological mechanisms with this structure:

1. Molecular/Cellular Level: What happens at the smallest scale?
2. System Level: How do organs/systems respond?
3. Clinical Effects: What are the observable outcomes?
4. Comparisons: How does this compare to alternatives?

Be specific about pathways, receptors, and measurable effects."""
        }
        
    async def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Answer a query using the knowledge base, conducting research if needed.
        This fixes the output format issue by enforcing answer-first structure.
        """
        # Check existing knowledge
        existing_knowledge = await self._search_knowledge_base(question)
        coverage = self._assess_knowledge_coverage(question, existing_knowledge)
        
        if coverage['score'] < 0.7:
            # Conduct new research
            logger.info(f"Knowledge gap detected (coverage: {coverage['score']:.2%}). Starting research...")
            research_results = await self._conduct_autonomous_research(question, coverage['gaps'])
            
            # Update knowledge base
            await self._update_knowledge_base(research_results)
            
            # Re-search with updated knowledge
            existing_knowledge = await self._search_knowledge_base(question)
            
        # Generate answer with proper format
        answer = await self._generate_formatted_answer(question, existing_knowledge)
        
        return {
            'answer': answer['response'],
            'confidence': answer['confidence'],
            'sources': answer['sources'],
            'research_conducted': coverage['score'] < 0.7,
            'knowledge_gaps_filled': coverage.get('gaps', [])
        }
        
    async def _search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search the vector knowledge base."""
        # Get query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search ChromaDB
        results = self.knowledge_collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        knowledge_items = []
        for i in range(len(results['documents'][0])):
            knowledge_items.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'relevance': 1 - results['distances'][0][i]  # Convert distance to relevance
            })
            
        return knowledge_items
        
    def _assess_knowledge_coverage(self, question: str, knowledge: List[Dict]) -> Dict[str, Any]:
        """Assess how well current knowledge covers the question."""
        # Extract key concepts from question
        concepts = self._extract_medical_concepts(question)
        
        # Check coverage for each concept
        covered = []
        gaps = []
        
        for concept in concepts:
            if self._is_concept_covered(concept, knowledge):
                covered.append(concept)
            else:
                gaps.append(concept)
                
        coverage_score = len(covered) / len(concepts) if concepts else 0
        
        return {
            'score': coverage_score,
            'covered_concepts': covered,
            'gaps': gaps,
            'total_concepts': len(concepts)
        }
        
    async def _conduct_autonomous_research(self, question: str, knowledge_gaps: List[str]) -> Dict[str, Any]:
        """Conduct autonomous research to fill knowledge gaps."""
        research_results = {
            'question': question,
            'gaps_addressed': knowledge_gaps,
            'sources': [],
            'findings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Search PubMed
        pubmed_results = await self._search_pubmed(question, knowledge_gaps)
        research_results['sources'].extend(pubmed_results)
        
        # Search ArXiv
        arxiv_results = await self._search_arxiv(question, knowledge_gaps)
        research_results['sources'].extend(arxiv_results)
        
        # Search ClinicalTrials.gov
        trial_results = await self._search_clinical_trials(question, knowledge_gaps)
        research_results['sources'].extend(trial_results)
        
        # Analyze and synthesize findings
        synthesis = await self._synthesize_research(research_results['sources'])
        research_results['findings'] = synthesis
        
        # Save research to library
        await self.library.save_document(
            f"auto_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "research_results",
            research_results
        )
        
        return research_results
        
    async def _search_pubmed(self, question: str, gaps: List[str]) -> List[Dict[str, Any]]:
        """Search PubMed for relevant papers."""
        results = []
        
        # Formulate search query
        search_terms = f"{question} {' '.join(gaps)}"
        
        try:
            # Search PubMed
            search_url = f"{self.pubmed_base}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': search_terms,
                'retmax': 10,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            
            if 'esearchresult' in search_data and 'idlist' in search_data['esearchresult']:
                pmids = search_data['esearchresult']['idlist']
                
                # Fetch abstracts
                if pmids:
                    fetch_url = f"{self.pubmed_base}efetch.fcgi"
                    fetch_params = {
                        'db': 'pubmed',
                        'id': ','.join(pmids),
                        'retmode': 'xml'
                    }
                    
                    fetch_response = requests.get(fetch_url, params=fetch_params)
                    
                    # Parse XML
                    root = ET.fromstring(fetch_response.content)
                    
                    for article in root.findall('.//PubmedArticle'):
                        title_elem = article.find('.//ArticleTitle')
                        abstract_elem = article.find('.//AbstractText')
                        pmid_elem = article.find('.//PMID')
                        
                        if title_elem is not None and abstract_elem is not None:
                            results.append({
                                'source': 'PubMed',
                                'pmid': pmid_elem.text if pmid_elem is not None else 'Unknown',
                                'title': title_elem.text,
                                'abstract': abstract_elem.text,
                                'relevance': 0.9,  # High relevance for PubMed
                                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text if pmid_elem is not None else ''}"
                            })
                            
        except Exception as e:
            logger.error(f"PubMed search error: {str(e)}")
            
        return results
        
    async def _search_arxiv(self, question: str, gaps: List[str]) -> List[Dict[str, Any]]:
        """Search ArXiv for relevant papers."""
        results = []
        
        try:
            # Search ArXiv
            search_query = f"{question} {' '.join(gaps)}"
            search = arxiv.Search(
                query=search_query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                results.append({
                    'source': 'ArXiv',
                    'arxiv_id': paper.entry_id,
                    'title': paper.title,
                    'abstract': paper.summary,
                    'authors': [author.name for author in paper.authors],
                    'relevance': 0.8,
                    'url': paper.entry_id,
                    'pdf_url': paper.pdf_url
                })
                
        except Exception as e:
            logger.error(f"ArXiv search error: {str(e)}")
            
        return results
        
    async def _search_clinical_trials(self, question: str, gaps: List[str]) -> List[Dict[str, Any]]:
        """Search ClinicalTrials.gov for relevant trials."""
        results = []
        
        try:
            # Formulate query
            query_terms = f"{question} {' '.join(gaps)}"
            
            params = {
                'expr': query_terms,
                'fields': 'NCTId,BriefTitle,BriefSummary,Condition,InterventionName',
                'min_rnk': 1,
                'max_rnk': 5,
                'fmt': 'json'
            }
            
            response = requests.get(self.clinicaltrials_base, params=params)
            data = response.json()
            
            if 'StudyFieldsResponse' in data:
                studies = data['StudyFieldsResponse']['StudyFields']
                
                for study in studies:
                    results.append({
                        'source': 'ClinicalTrials.gov',
                        'nct_id': study.get('NCTId', ['Unknown'])[0],
                        'title': study.get('BriefTitle', ['Unknown'])[0],
                        'summary': study.get('BriefSummary', ['No summary'])[0],
                        'conditions': study.get('Condition', []),
                        'interventions': study.get('InterventionName', []),
                        'relevance': 0.7,
                        'url': f"https://clinicaltrials.gov/ct2/show/{study.get('NCTId', [''])[0]}"
                    })
                    
        except Exception as e:
            logger.error(f"ClinicalTrials.gov search error: {str(e)}")
            
        return results
        
    async def _synthesize_research(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synthesize findings from multiple sources."""
        synthesis = []
        
        # Group by topic/intervention
        grouped = {}
        for source in sources:
            # Extract key topics (simplified - in production use NER)
            key = source.get('title', '').lower()[:50]
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(source)
            
        # Synthesize each group
        for topic, topic_sources in grouped.items():
            synthesis.append({
                'topic': topic,
                'source_count': len(topic_sources),
                'consensus': self._determine_consensus(topic_sources),
                'key_findings': self._extract_key_findings(topic_sources),
                'confidence': self._calculate_confidence(topic_sources)
            })
            
        return synthesis
        
    async def _update_knowledge_base(self, research_results: Dict[str, Any]):
        """Update the vector knowledge base with new findings."""
        # Process each source
        for source in research_results['sources']:
            # Create document text
            doc_text = f"{source.get('title', '')} {source.get('abstract', '')} {source.get('summary', '')}"
            
            # Generate embedding
            embedding = self.embedder.encode(doc_text).tolist()
            
            # Add to ChromaDB
            self.knowledge_collection.add(
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[{
                    'source_type': source['source'],
                    'title': source.get('title', ''),
                    'url': source.get('url', ''),
                    'added_date': datetime.now().isoformat(),
                    'relevance': source.get('relevance', 0.5)
                }],
                ids=[f"{source['source']}_{source.get('pmid', source.get('arxiv_id', source.get('nct_id', 'unknown')))}"]
            )
            
        logger.info(f"Updated knowledge base with {len(research_results['sources'])} new sources")
        
    async def _generate_formatted_answer(self, question: str, knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate properly formatted answer using the answer-first prompt."""
        # Prepare context from knowledge
        context = self._prepare_context(knowledge)
        
        # Create the prompt that enforces answer-first format
        prompt = f"""{self.system_prompts['answer_first']}

Question: {question}

Context from knowledge base:
{context}

Remember: Provide your ANSWER first, then list SOURCES at the end."""
        
        # Generate response
        response = await self.chat.generate(prompt)
        
        # Extract answer and sources
        answer_parts = self._parse_formatted_response(response)
        
        # Calculate confidence based on knowledge coverage
        confidence = min(0.95, sum(item['relevance'] for item in knowledge[:5]) / 5) if knowledge else 0.3
        
        return {
            'response': answer_parts['answer'],
            'sources': answer_parts['sources'],
            'confidence': confidence,
            'raw_response': response
        }
        
    def _prepare_context(self, knowledge: List[Dict[str, Any]]) -> str:
        """Prepare context from knowledge items."""
        context_parts = []
        
        for i, item in enumerate(knowledge[:5], 1):  # Top 5 most relevant
            context_parts.append(f"""
Item {i} (Relevance: {item['relevance']:.2f}):
{item['content'][:500]}...
Source: {item['metadata'].get('source_type', 'Unknown')}
""")
        
        return "\n".join(context_parts)
        
    def _parse_formatted_response(self, response: str) -> Dict[str, str]:
        """Parse the formatted response to extract answer and sources."""
        parts = {
            'answer': '',
            'sources': []
        }
        
        # Split by sections
        sections = response.split('[SOURCES]')
        
        if len(sections) >= 2:
            # Extract answer (remove [ANSWER] marker if present)
            answer_text = sections[0].replace('[ANSWER]', '').strip()
            parts['answer'] = answer_text
            
            # Extract sources
            sources_text = sections[1].strip()
            if sources_text:
                # Parse individual sources
                source_lines = sources_text.split('\n')
                for line in source_lines:
                    if line.strip() and not line.startswith('(') and not line.startswith('['):
                        parts['sources'].append(line.strip())
        else:
            # Fallback if markers not found
            parts['answer'] = response
            
        return parts
        
    def _extract_medical_concepts(self, text: str) -> List[str]:
        """Extract medical concepts from text."""
        # Simplified concept extraction
        # In production, use BioBERT or scispaCy
        concepts = []
        
        # Common medical terms to look for
        medical_terms = [
            'drug', 'medication', 'treatment', 'therapy', 'disease', 'condition',
            'symptom', 'side effect', 'mechanism', 'pathway', 'receptor', 'enzyme',
            'protein', 'gene', 'clinical', 'trial', 'efficacy', 'safety'
        ]
        
        text_lower = text.lower()
        for term in medical_terms:
            if term in text_lower:
                concepts.append(term)
                
        # Also extract specific drug/condition names (simplified)
        # In production, use NER
        if 'creatine' in text_lower:
            concepts.append('creatine')
        if 'alternative' in text_lower:
            concepts.append('alternative_treatments')
            
        return list(set(concepts))  # Remove duplicates
        
    def _is_concept_covered(self, concept: str, knowledge: List[Dict]) -> bool:
        """Check if a concept is covered in the knowledge base."""
        for item in knowledge:
            if concept.lower() in item['content'].lower():
                return True
        return False
        
    def _determine_consensus(self, sources: List[Dict]) -> str:
        """Determine consensus among sources."""
        # Simplified consensus determination
        if len(sources) < 2:
            return "Limited evidence"
        elif len(sources) < 5:
            return "Emerging evidence"
        else:
            return "Strong evidence base"
            
    def _extract_key_findings(self, sources: List[Dict]) -> List[str]:
        """Extract key findings from sources."""
        findings = []
        
        for source in sources[:3]:  # Top 3 sources
            # Extract first sentence of abstract/summary as key finding
            text = source.get('abstract', source.get('summary', ''))
            if text:
                first_sentence = text.split('.')[0] + '.'
                findings.append(first_sentence)
                
        return findings
        
    def _calculate_confidence(self, sources: List[Dict]) -> float:
        """Calculate confidence score based on sources."""
        if not sources:
            return 0.0
            
        # Factors: number of sources, relevance scores, source types
        base_confidence = min(len(sources) / 10, 0.5)  # Max 0.5 from count
        
        # Add relevance scores
        avg_relevance = sum(s.get('relevance', 0.5) for s in sources) / len(sources)
        relevance_bonus = avg_relevance * 0.3
        
        # Bonus for diverse source types
        source_types = set(s['source'] for s in sources)
        diversity_bonus = min(len(source_types) / 3, 0.2)  # Max 0.2 for 3+ types
        
        return min(base_confidence + relevance_bonus + diversity_bonus, 0.95)
        
    async def find_alternatives(self, compound: str, requirements: List[str] = None) -> Dict[str, Any]:
        """
        Find alternatives to a given compound/treatment.
        Example: find_alternatives("creatine", ["legal", "better", "same effect"])
        """
        # Formulate research question
        req_str = " and ".join(requirements) if requirements else ""
        question = f"What are alternatives to {compound} that are {req_str}?"
        
        # Use the main answer_query method which handles research
        result = await self.answer_query(question)
        
        # Extract specific alternatives from the answer
        alternatives = self._extract_alternatives(result['answer'])
        
        return {
            'original': compound,
            'requirements': requirements,
            'alternatives': alternatives,
            'full_analysis': result['answer'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'research_conducted': result['research_conducted']
        }
        
    def _extract_alternatives(self, answer_text: str) -> List[Dict[str, str]]:
        """Extract specific alternatives mentioned in the answer."""
        alternatives = []
        
        # Look for common patterns mentioning alternatives
        # This is simplified - in production use NER
        lines = answer_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in ['alternative', 'instead', 'similar to', 'like']):
                # Extract the alternative name (simplified)
                alternatives.append({
                    'name': line.strip(),
                    'mentioned_in_context': line
                })
                
        return alternatives[:5]  # Top 5 alternatives


class MedicalKnowledgeValidator:
    """Validates and scores medical knowledge for confidence and accuracy."""
    
    def __init__(self):
        self.trusted_sources = ['PubMed', 'Cochrane', 'FDA', 'NIH', 'WHO']
        
    def validate_claim(self, claim: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a medical claim against sources."""
        validation = {
            'claim': claim,
            'validity_score': 0.0,
            'confidence': 0.0,
            'supporting_sources': [],
            'contradicting_sources': [],
            'assessment': ''
        }
        
        # Check each source
        for source in sources:
            relevance = source.get('relevance', 0.5)
            if self._supports_claim(claim, source):
                validation['supporting_sources'].append(source)
                validation['validity_score'] += relevance
            elif self._contradicts_claim(claim, source):
                validation['contradicting_sources'].append(source)
                validation['validity_score'] -= relevance * 0.5
                
        # Calculate confidence
        total_sources = len(validation['supporting_sources']) + len(validation['contradicting_sources'])
        if total_sources > 0:
            validation['confidence'] = min(total_sources / 10, 0.9)
            
        # Determine assessment
        if validation['validity_score'] > 0.7:
            validation['assessment'] = 'Strongly supported'
        elif validation['validity_score'] > 0.3:
            validation['assessment'] = 'Moderately supported'
        elif validation['validity_score'] > 0:
            validation['assessment'] = 'Weakly supported'
        else:
            validation['assessment'] = 'Not supported or contradicted'
            
        return validation
        
    def _supports_claim(self, claim: str, source: Dict[str, Any]) -> bool:
        """Check if source supports the claim."""
        # Simplified - in production use NLP similarity
        claim_lower = claim.lower()
        source_text = f"{source.get('title', '')} {source.get('abstract', '')}".lower()
        
        # Look for affirmative patterns
        return any(pattern in source_text for pattern in ['supports', 'confirms', 'demonstrates', 'shows that'])
        
    def _contradicts_claim(self, claim: str, source: Dict[str, Any]) -> bool:
        """Check if source contradicts the claim."""
        # Simplified - in production use NLP
        source_text = f"{source.get('title', '')} {source.get('abstract', '')}".lower()
        
        # Look for contradictory patterns
        return any(pattern in source_text for pattern in ['contradicts', 'refutes', 'no evidence', 'not supported']) 