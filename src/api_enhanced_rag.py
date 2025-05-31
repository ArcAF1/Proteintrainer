"""
API-Enhanced RAG - Combines local knowledge with real-time API data
Optimized for M1 MacBook with intelligent caching and source prioritization
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from .rag_chat import RAGChat
from .biomedical_api_client import BiomedicalAPIClient
from .api_integration_config import API_CONFIGS

logger = logging.getLogger(__name__)


class APIEnhancedRAG(RAGChat):
    """RAG system enhanced with real-time API data."""
    
    def __init__(self, use_apis: bool = True, cache_hours: int = 24):
        super().__init__()
        self.use_apis = use_apis
        self.cache_hours = cache_hours
        self.api_client = None
        self._api_cache = {}
        self._cache_timestamps = {}
        
    async def _ensure_api_client(self):
        """Ensure API client is initialized."""
        if self.api_client is None and self.use_apis:
            self.api_client = BiomedicalAPIClient()
            # Create a persistent session
            self.api_client.session = await self.api_client.session.__aenter__()
            
    def _is_api_cache_valid(self, query: str) -> bool:
        """Check if cached API data is still valid."""
        if query not in self._cache_timestamps:
            return False
            
        age = datetime.now() - self._cache_timestamps[query]
        return age < timedelta(hours=self.cache_hours)
        
    async def _get_api_context(self, query: str) -> Dict[str, Any]:
        """Get relevant context from APIs."""
        # Check cache first
        if self._is_api_cache_valid(query):
            logger.info(f"Using cached API data for: {query}")
            return self._api_cache.get(query, {})
            
        await self._ensure_api_client()
        
        # Determine which APIs to query based on the question
        api_results = {}
        
        # Keywords that suggest we need clinical evidence
        clinical_keywords = ["trial", "study", "evidence", "effective", "works", "proven"]
        if any(keyword in query.lower() for keyword in clinical_keywords):
            logger.info("Querying clinical trials API...")
            trials = await self.api_client.search_clinical_trials(query, max_results=5)
            api_results["clinical_trials"] = trials
            
        # Keywords that suggest we need recent research
        research_keywords = ["latest", "recent", "new", "current", "2024", "2023"]
        if any(keyword in query.lower() for keyword in research_keywords):
            logger.info("Querying recent publications...")
            pmc_results = await self.api_client.search_europe_pmc(query, max_results=10)
            api_results["recent_papers"] = pmc_results
            
        # Keywords that suggest we need compound/drug information
        compound_keywords = ["creatine", "supplement", "drug", "compound", "molecule"]
        if any(keyword in query.lower() for keyword in compound_keywords):
            # Extract compound name (simplified)
            for keyword in compound_keywords:
                if keyword in query.lower() and keyword != "supplement":
                    logger.info(f"Getting compound info for: {keyword}")
                    compound_info = await self.api_client.get_compound_info(keyword)
                    if compound_info:
                        api_results["compound_info"] = compound_info
                    break
                    
        # Cache the results
        self._api_cache[query] = api_results
        self._cache_timestamps[query] = datetime.now()
        
        return api_results
        
    def _format_api_context(self, api_results: Dict[str, Any]) -> str:
        """Format API results into context text."""
        context_parts = []
        
        # Format clinical trials
        if "clinical_trials" in api_results and api_results["clinical_trials"]:
            context_parts.append("**Recent Clinical Trials:**")
            for trial in api_results["clinical_trials"][:3]:
                protocol = trial.get("protocolSection", {})
                title = protocol.get("identificationModule", {}).get("briefTitle", "Unknown")
                status = protocol.get("statusModule", {}).get("overallStatus", "Unknown")
                nct_id = protocol.get("identificationModule", {}).get("nctId", "")
                
                context_parts.append(f"- {title} (Status: {status}, ID: {nct_id})")
                
        # Format recent papers
        if "recent_papers" in api_results and api_results["recent_papers"]:
            context_parts.append("\n**Recent Research Papers:**")
            for paper in api_results["recent_papers"][:3]:
                title = paper.get("title", "Unknown")
                year = paper.get("pubYear", "Unknown")
                pmid = paper.get("pmid", "")
                
                context_parts.append(f"- {title} ({year}) [PMID: {pmid}]")
                
                # Include abstract snippet if available
                abstract = paper.get("abstractText", "")
                if abstract:
                    snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
                    context_parts.append(f"  {snippet}")
                    
        # Format compound info
        if "compound_info" in api_results and api_results["compound_info"]:
            compound = api_results["compound_info"]
            context_parts.append("\n**Compound Information:**")
            
            # Extract basic properties
            props = compound.get("props", [])
            for prop in props[:5]:  # Limit properties shown
                if "urn" in prop and "value" in prop:
                    label = prop["urn"].get("label", "Unknown")
                    value = prop["value"].get("sval") or prop["value"].get("fval", "")
                    if value:
                        context_parts.append(f"- {label}: {value}")
                        
        return "\n".join(context_parts)
        
    async def answer(self, question: str) -> str:
        """Answer with both local knowledge and API data."""
        if not self.is_ready():
            return await super().answer(question)
            
        try:
            # Get local context first
            local_docs = self.retrieve(question)
            
            # Get API context if enabled
            api_context = ""
            if self.use_apis:
                try:
                    api_results = await self._get_api_context(question)
                    if api_results:
                        api_context = self._format_api_context(api_results)
                except Exception as e:
                    logger.error(f"API enhancement failed: {e}")
                    # Continue with just local data
                    
            # Build enhanced prompt
            context_parts = []
            
            # Add API context first (if available)
            if api_context:
                context_parts.append("**Live Data from Scientific APIs:**\n" + api_context)
                
            # Add local context
            if local_docs:
                context_parts.append("\n**From Local Knowledge Base:**")
                for idx, doc in enumerate(local_docs[:3], start=1):
                    truncated = doc[:500] + "..." if len(doc) > 500 else doc
                    context_parts.append(f"[{idx}] {truncated}")
                    
            # Combine all context
            full_context = "\n\n".join(context_parts)
            
            # Create enhanced prompt
            prompt = f"""You are a biomedical AI assistant with access to both a local knowledge base and real-time scientific data.

Based on the following context, please answer the user's question comprehensively.

Context:
{full_context}

User Question: {question}

Instructions:
1. Synthesize information from both local knowledge and live API data
2. Cite specific clinical trials by NCT ID when relevant
3. Reference recent papers by PMID when applicable
4. Provide practical, evidence-based answers
5. Note if information comes from recent studies vs established knowledge

Answer:"""

            # Generate response
            response = await self.generate(prompt)
            
            # Add source attribution
            sources_used = []
            if api_context:
                sources_used.append("Live scientific APIs")
            if local_docs:
                sources_used.append("Local knowledge base")
                
            if sources_used:
                response += f"\n\n**Sources:** {', '.join(sources_used)}"
                
            return response
            
        except Exception as e:
            logger.error(f"Enhanced answer failed: {e}")
            # Fallback to standard RAG
            return await super().answer(question)
            
    async def close(self):
        """Clean up API client resources."""
        if self.api_client and self.api_client.session:
            await self.api_client.session.close()
            

# Enhanced answer function for direct use
async def api_enhanced_answer(question: str, use_apis: bool = True) -> str:
    """Answer using API-enhanced RAG."""
    rag = APIEnhancedRAG(use_apis=use_apis)
    try:
        return await rag.answer(question)
    finally:
        await rag.close()
        

# Example usage
async def example_enhanced_rag():
    """Example of using API-enhanced RAG."""
    questions = [
        "What are the latest clinical trials on creatine for muscle performance?",
        "What is the current evidence for creatine absorption enhancement?",
        "Tell me about creatine's chemical properties and mechanisms"
    ]
    
    rag = APIEnhancedRAG(use_apis=True)
    
    try:
        for question in questions:
            print(f"\nQuestion: {question}")
            print("-" * 80)
            
            answer = await rag.answer(question)
            print(answer)
            print()
            
    finally:
        await rag.close()
        

if __name__ == "__main__":
    asyncio.run(example_enhanced_rag()) 