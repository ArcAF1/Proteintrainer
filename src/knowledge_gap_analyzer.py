"""
Knowledge Gap Analyzer for Biomedical AI System
Identifies what information is missing and suggests data sources
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class KnowledgeGapAnalyzer:
    """Analyzes knowledge gaps in the biomedical AI system."""
    
    def __init__(self):
        self.knowledge_map = defaultdict(set)
        self.data_sources = {}
        self.query_history = []
        self.identified_gaps = []
        
        self._load_knowledge_schema()
        
    def _load_knowledge_schema(self):
        """Load schema of expected biomedical knowledge."""
        self.knowledge_schema = {
            'drugs': {
                'required_fields': [
                    'mechanism_of_action',
                    'targets',
                    'indications',
                    'contraindications',
                    'interactions',
                    'pharmacokinetics',
                    'adverse_effects',
                    'dosing'
                ],
                'data_sources': [
                    'DrugBank',
                    'PubChem',
                    'ChEMBL',
                    'FDA Orange Book',
                    'Clinical Trials'
                ]
            },
            'diseases': {
                'required_fields': [
                    'pathophysiology',
                    'symptoms',
                    'diagnosis',
                    'treatment_options',
                    'prognosis',
                    'epidemiology',
                    'genetic_factors'
                ],
                'data_sources': [
                    'OMIM',
                    'DisGeNET',
                    'MedGen',
                    'Orphanet',
                    'PubMed'
                ]
            },
            'proteins': {
                'required_fields': [
                    'structure',
                    'function',
                    'interactions',
                    'pathways',
                    'expression',
                    'mutations',
                    'drug_targets'
                ],
                'data_sources': [
                    'UniProt',
                    'PDB',
                    'STRING',
                    'Reactome',
                    'KEGG'
                ]
            },
            'pathways': {
                'required_fields': [
                    'components',
                    'regulation',
                    'disease_associations',
                    'drug_targets',
                    'cross_talk'
                ],
                'data_sources': [
                    'Reactome',
                    'KEGG',
                    'WikiPathways',
                    'BioCyc'
                ]
            }
        }
        
    def analyze_query_coverage(self, query: str, available_docs: List[str]) -> Dict[str, Any]:
        """Analyze how well available knowledge covers a query."""
        
        # Identify query type and requirements
        query_analysis = self._analyze_query_requirements(query)
        
        # Check coverage
        coverage = self._check_knowledge_coverage(query_analysis, available_docs)
        
        # Identify specific gaps
        gaps = self._identify_specific_gaps(query_analysis, coverage)
        
        # Suggest data sources
        suggestions = self._suggest_data_sources(gaps)
        
        # Store for learning
        self.query_history.append({
            'query': query,
            'coverage': coverage,
            'gaps': gaps
        })
        
        return {
            'query_type': query_analysis['type'],
            'required_knowledge': query_analysis['requirements'],
            'coverage_score': coverage['score'],
            'covered_aspects': coverage['covered'],
            'missing_aspects': coverage['missing'],
            'specific_gaps': gaps,
            'suggested_sources': suggestions,
            'confidence': self._calculate_confidence(coverage)
        }
        
    def _analyze_query_requirements(self, query: str) -> Dict[str, Any]:
        """Analyze what knowledge is required to answer the query."""
        query_lower = query.lower()
        
        # Determine primary entity type
        entity_type = None
        if any(term in query_lower for term in ['drug', 'medication', 'compound']):
            entity_type = 'drugs'
        elif any(term in query_lower for term in ['disease', 'condition', 'disorder']):
            entity_type = 'diseases'
        elif any(term in query_lower for term in ['protein', 'enzyme', 'receptor']):
            entity_type = 'proteins'
        elif any(term in query_lower for term in ['pathway', 'signaling', 'cascade']):
            entity_type = 'pathways'
            
        # Identify specific requirements
        requirements = []
        if entity_type:
            schema = self.knowledge_schema.get(entity_type, {})
            required_fields = schema.get('required_fields', [])
            
            # Check which fields are likely needed based on query
            for field in required_fields:
                if self._is_field_relevant_to_query(field, query_lower):
                    requirements.append(field)
                    
        return {
            'type': entity_type or 'general',
            'requirements': requirements or ['general_information']
        }
        
    def _is_field_relevant_to_query(self, field: str, query_lower: str) -> bool:
        """Check if a knowledge field is relevant to the query."""
        field_keywords = {
            'mechanism_of_action': ['how', 'work', 'mechanism', 'action'],
            'targets': ['target', 'bind', 'receptor', 'enzyme'],
            'interactions': ['interact', 'combination', 'with'],
            'adverse_effects': ['side effect', 'adverse', 'safety', 'risk'],
            'pharmacokinetics': ['absorption', 'metabolism', 'elimination', 'half-life'],
            'pathophysiology': ['cause', 'mechanism', 'pathology'],
            'treatment_options': ['treat', 'therapy', 'management']
        }
        
        keywords = field_keywords.get(field, [field.replace('_', ' ')])
        return any(keyword in query_lower for keyword in keywords)
        
    def _check_knowledge_coverage(
        self, 
        query_analysis: Dict, 
        available_docs: List[str]
    ) -> Dict[str, Any]:
        """Check how well available documents cover the requirements."""
        covered = []
        missing = []
        
        all_docs_text = ' '.join(available_docs).lower()
        
        for requirement in query_analysis['requirements']:
            if self._is_requirement_covered(requirement, all_docs_text):
                covered.append(requirement)
            else:
                missing.append(requirement)
                
        score = len(covered) / len(query_analysis['requirements']) if query_analysis['requirements'] else 0
        
        return {
            'score': score,
            'covered': covered,
            'missing': missing
        }
        
    def _is_requirement_covered(self, requirement: str, docs_text: str) -> bool:
        """Check if a requirement is covered in the documents."""
        # Simplified check - in production, use NLP
        requirement_terms = requirement.replace('_', ' ').split()
        return any(term in docs_text for term in requirement_terms)
        
    def _identify_specific_gaps(
        self, 
        query_analysis: Dict,
        coverage: Dict
    ) -> List[Dict[str, Any]]:
        """Identify specific knowledge gaps."""
        gaps = []
        
        for missing_aspect in coverage['missing']:
            gap = {
                'aspect': missing_aspect,
                'importance': self._assess_importance(missing_aspect, query_analysis),
                'description': self._describe_gap(missing_aspect),
                'impact': self._assess_impact(missing_aspect)
            }
            gaps.append(gap)
            
        return sorted(gaps, key=lambda x: x['importance'], reverse=True)
        
    def _assess_importance(self, aspect: str, query_analysis: Dict) -> float:
        """Assess importance of a missing aspect (0-1)."""
        # Critical aspects for each entity type
        critical_aspects = {
            'drugs': ['mechanism_of_action', 'adverse_effects', 'interactions'],
            'diseases': ['pathophysiology', 'treatment_options'],
            'proteins': ['function', 'structure', 'drug_targets']
        }
        
        entity_type = query_analysis['type']
        if entity_type in critical_aspects:
            if aspect in critical_aspects[entity_type]:
                return 0.9
                
        return 0.5  # Default importance
        
    def _describe_gap(self, aspect: str) -> str:
        """Generate human-readable description of the gap."""
        descriptions = {
            'mechanism_of_action': "How the drug works at a molecular level",
            'adverse_effects': "Potential side effects and safety concerns",
            'interactions': "Drug-drug and drug-food interactions",
            'pathophysiology': "The biological mechanisms causing the disease",
            'treatment_options': "Available therapeutic approaches",
            'pharmacokinetics': "How the body processes the drug (ADME)",
            'targets': "Molecular targets the drug binds to",
            'structure': "3D structure and conformational details"
        }
        
        return descriptions.get(aspect, f"Information about {aspect.replace('_', ' ')}")
        
    def _assess_impact(self, aspect: str) -> str:
        """Assess impact of missing this information."""
        high_impact_aspects = [
            'mechanism_of_action', 
            'adverse_effects', 
            'interactions',
            'contraindications'
        ]
        
        if aspect in high_impact_aspects:
            return "high"
        else:
            return "moderate"
            
    def _suggest_data_sources(self, gaps: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest data sources to fill the gaps."""
        suggestions = []
        suggested_sources = set()
        
        for gap in gaps:
            aspect = gap['aspect']
            
            # Find relevant data sources
            for entity_type, schema in self.knowledge_schema.items():
                if aspect in schema.get('required_fields', []):
                    sources = schema.get('data_sources', [])
                    
                    for source in sources:
                        if source not in suggested_sources:
                            suggestion = {
                                'source': source,
                                'reason': f"Contains {gap['description']}",
                                'priority': 'high' if gap['importance'] > 0.7 else 'medium',
                                'url': self._get_source_url(source)
                            }
                            suggestions.append(suggestion)
                            suggested_sources.add(source)
                            
        return suggestions
        
    def _get_source_url(self, source: str) -> str:
        """Get URL for a data source."""
        urls = {
            'DrugBank': 'https://www.drugbank.ca/',
            'PubChem': 'https://pubchem.ncbi.nlm.nih.gov/',
            'ChEMBL': 'https://www.ebi.ac.uk/chembl/',
            'UniProt': 'https://www.uniprot.org/',
            'Reactome': 'https://reactome.org/',
            'KEGG': 'https://www.kegg.jp/',
            'Clinical Trials': 'https://clinicaltrials.gov/'
        }
        
        return urls.get(source, f"https://www.google.com/search?q={source}")
        
    def _calculate_confidence(self, coverage: Dict) -> float:
        """Calculate confidence in ability to answer based on coverage."""
        base_confidence = coverage['score']
        
        # Adjust based on missing critical information
        critical_missing = sum(1 for aspect in coverage['missing'] 
                              if aspect in ['mechanism_of_action', 'adverse_effects'])
        
        confidence = base_confidence - (critical_missing * 0.2)
        return max(0.1, min(1.0, confidence))
        
    def generate_knowledge_report(self) -> Dict[str, Any]:
        """Generate a report of overall knowledge gaps."""
        
        # Aggregate gaps from history
        all_gaps = defaultdict(int)
        for entry in self.query_history:
            for gap in entry['gaps']:
                all_gaps[gap['aspect']] += 1
                
        # Sort by frequency
        common_gaps = sorted(
            all_gaps.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'total_queries_analyzed': len(self.query_history),
            'most_common_gaps': common_gaps[:10],
            'recommended_data_sources': self._recommend_priority_sources(common_gaps),
            'coverage_trend': self._calculate_coverage_trend()
        }
        
    def _recommend_priority_sources(self, common_gaps: List[Tuple[str, int]]) -> List[str]:
        """Recommend priority data sources based on common gaps."""
        recommended = set()
        
        for gap_aspect, _ in common_gaps[:5]:
            for entity_type, schema in self.knowledge_schema.items():
                if gap_aspect in schema.get('required_fields', []):
                    recommended.update(schema.get('data_sources', [])[:2])
                    
        return list(recommended)
        
    def _calculate_coverage_trend(self) -> Dict[str, float]:
        """Calculate coverage trend over queries."""
        if len(self.query_history) < 2:
            return {'trend': 'insufficient_data'}
            
        recent_coverage = [
            entry['coverage']['score'] 
            for entry in self.query_history[-10:]
        ]
        
        avg_coverage = sum(recent_coverage) / len(recent_coverage)
        
        return {
            'average_coverage': avg_coverage,
            'trend': 'improving' if recent_coverage[-1] > recent_coverage[0] else 'stable'
        } 