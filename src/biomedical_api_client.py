"""
Biomedical API Client - Unified interface for fetching scientific data
Optimized for M1 MacBook with rate limiting and caching
"""
from __future__ import annotations

import asyncio
import aiohttp
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib

from .api_integration_config import API_CONFIGS, APIConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, rate_limit: float):
        self.rate_limit = rate_limit
        self.min_interval = 1.0 / rate_limit
        self.last_call = 0.0
        
    async def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        time_since_last = now - self.last_call
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        self.last_call = time.time()


class BiomedicalAPIClient:
    """Unified client for biomedical APIs with caching and rate limiting."""
    
    def __init__(self, cache_dir: str = "data/api_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiters = {
            name: RateLimiter(config.rate_limit)
            for name, config in API_CONFIGS.items()
        }
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    def _get_cache_path(self, api_name: str, query: str) -> Path:
        """Generate cache file path for a query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.cache_dir / f"{api_name}_{query_hash}.json"
        
    def _is_cache_valid(self, cache_path: Path, max_age_days: int = 7) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(days=max_age_days)
        
    async def _fetch_with_cache(self, api_name: str, url: str, 
                               use_cache: bool = True,
                               cache_days: int = 7) -> Dict[str, Any]:
        """Fetch data with caching support."""
        cache_path = self._get_cache_path(api_name, url)
        
        # Check cache first
        if use_cache and self._is_cache_valid(cache_path, cache_days):
            logger.info(f"Using cached data for {api_name}")
            with open(cache_path, 'r') as f:
                return json.load(f)
                
        # Fetch fresh data
        logger.info(f"Fetching fresh data from {api_name}")
        await self.rate_limiters[api_name].wait()
        
        async with self.session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            
        # Cache the response
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return data
        
    # PubMed Methods
    async def search_pubmed(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed and return PMIDs."""
        config = API_CONFIGS["pubmed"]
        url = f"{config.base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
        
        data = await self._fetch_with_cache("pubmed", url)
        return data.get("esearchresult", {}).get("idlist", [])
        
    async def fetch_pubmed_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch abstracts for given PMIDs."""
        if not pmids:
            return []
            
        config = API_CONFIGS["pubmed"]
        pmid_str = ",".join(pmids[:200])  # Limit to 200 at a time
        url = f"{config.base_url}efetch.fcgi?db=pubmed&id={pmid_str}&retmode=xml"
        
        # For XML, we need to handle differently
        await self.rate_limiters["pubmed"].wait()
        async with self.session.get(url) as response:
            xml_data = await response.text()
            
        # Parse XML (simplified - in production use proper XML parser)
        articles = []
        # This is a placeholder - implement proper XML parsing
        logger.warning("XML parsing not implemented - returning empty list")
        return articles
        
    # Europe PMC Methods
    async def search_europe_pmc(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search Europe PMC for full-text articles."""
        config = API_CONFIGS["europe_pmc"]
        url = f"{config.base_url}search?query={query}&format=json&pageSize={max_results}"
        
        data = await self._fetch_with_cache("europe_pmc", url)
        return data.get("resultList", {}).get("result", [])
        
    # Clinical Trials Methods
    async def search_clinical_trials(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search ClinicalTrials.gov for relevant studies."""
        config = API_CONFIGS["clinicaltrials"]
        url = f"{config.base_url}studies?query.term={query}&pageSize={max_results}"
        
        data = await self._fetch_with_cache("clinicaltrials", url)
        return data.get("studies", [])
        
    # PubChem Methods
    async def get_compound_info(self, compound_name: str) -> Dict[str, Any]:
        """Get compound information from PubChem."""
        config = API_CONFIGS["pubchem"]
        url = f"{config.base_url}compound/name/{compound_name}/JSON"
        
        try:
            data = await self._fetch_with_cache("pubchem", url)
            return data.get("PC_Compounds", [{}])[0] if data.get("PC_Compounds") else {}
        except Exception as e:
            logger.error(f"Error fetching PubChem data: {e}")
            return {}
            
    # MeSH Methods
    async def search_mesh(self, term: str) -> List[Dict[str, Any]]:
        """Search MeSH for term definitions and hierarchy."""
        config = API_CONFIGS["mesh"]
        url = f"{config.base_url}lookup/descriptor?label={term}&match=contains&limit=10"
        
        try:
            # MeSH returns JSON-LD
            await self.rate_limiters["mesh"].wait()
            async with self.session.get(url, headers={"Accept": "application/json"}) as response:
                data = await response.json()
                return data
        except Exception as e:
            logger.error(f"Error fetching MeSH data: {e}")
            return []
            
    # Unified search across multiple APIs
    async def search_all_sources(self, query: str, 
                                sources: List[str] = None,
                                max_results_per_source: int = 50) -> Dict[str, Any]:
        """Search multiple sources simultaneously."""
        if sources is None:
            sources = ["pubmed", "europe_pmc", "clinicaltrials"]
            
        results = {}
        tasks = []
        
        if "pubmed" in sources:
            tasks.append(("pubmed", self.search_pubmed(query, max_results_per_source)))
        if "europe_pmc" in sources:
            tasks.append(("europe_pmc", self.search_europe_pmc(query, max_results_per_source)))
        if "clinicaltrials" in sources:
            tasks.append(("clinicaltrials", self.search_clinical_trials(query, max_results_per_source)))
            
        # Execute searches in parallel
        for source, task in tasks:
            try:
                results[source] = await task
            except Exception as e:
                logger.error(f"Error searching {source}: {e}")
                results[source] = []
                
        return results
        
    # Methods for Neo4j integration
    def format_for_neo4j(self, source: str, data: Any) -> List[Dict[str, Any]]:
        """Format API data for Neo4j import."""
        formatted = []
        
        if source == "pubmed":
            # Format PubMed articles
            for pmid in data:
                formatted.append({
                    "type": "Publication",
                    "properties": {
                        "pmid": pmid,
                        "source": "PubMed"
                    }
                })
                
        elif source == "europe_pmc":
            # Format Europe PMC articles
            for article in data:
                formatted.append({
                    "type": "Publication",
                    "properties": {
                        "pmid": article.get("pmid"),
                        "pmcid": article.get("pmcid"),
                        "title": article.get("title"),
                        "abstract": article.get("abstractText"),
                        "doi": article.get("doi"),
                        "source": "EuropePMC"
                    }
                })
                
        elif source == "clinicaltrials":
            # Format clinical trials
            for trial in data:
                protocol = trial.get("protocolSection", {})
                formatted.append({
                    "type": "ClinicalTrial",
                    "properties": {
                        "nct_id": protocol.get("identificationModule", {}).get("nctId"),
                        "title": protocol.get("identificationModule", {}).get("briefTitle"),
                        "status": protocol.get("statusModule", {}).get("overallStatus"),
                        "phase": protocol.get("designModule", {}).get("phases", []),
                        "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
                        "interventions": [
                            int_data.get("name") 
                            for int_data in protocol.get("armsInterventionsModule", {}).get("interventions", [])
                        ]
                    }
                })
                
        return formatted
        
    # Creatine-specific convenience methods
    async def get_creatine_research(self) -> Dict[str, Any]:
        """Get comprehensive creatine research data."""
        queries = [
            "creatine supplementation muscle",
            "creatine monohydrate performance",
            "creatine absorption enhancement",
            "creatine timing protocol"
        ]
        
        all_results = {}
        for query in queries:
            results = await self.search_all_sources(query, max_results_per_source=25)
            for source, data in results.items():
                if source not in all_results:
                    all_results[source] = []
                all_results[source].extend(data)
                
        return all_results
        
    async def get_supplement_interactions(self, supplement: str = "creatine") -> Dict[str, Any]:
        """Get supplement interaction data."""
        # This would integrate with SUPP.AI when implemented
        logger.info(f"Getting interactions for {supplement}")
        # Placeholder for now
        return {
            "supplement": supplement,
            "interactions": [],
            "evidence": []
        }


# Example usage function
async def example_api_usage():
    """Example of how to use the API client."""
    async with BiomedicalAPIClient() as client:
        # Search for creatine studies
        print("Searching for creatine studies...")
        results = await client.search_all_sources("creatine muscle performance")
        
        print(f"\nFound {len(results['pubmed'])} PubMed articles")
        print(f"Found {len(results['europe_pmc'])} Europe PMC articles")
        print(f"Found {len(results['clinicaltrials'])} clinical trials")
        
        # Get compound info
        print("\nGetting creatine compound info...")
        compound_info = await client.get_compound_info("creatine")
        if compound_info:
            print(f"PubChem CID: {compound_info.get('id', {}).get('id', {}).get('cid')}")
            
        # Format for Neo4j
        print("\nFormatting for Neo4j...")
        neo4j_data = []
        for source, data in results.items():
            neo4j_data.extend(client.format_for_neo4j(source, data))
            
        print(f"Prepared {len(neo4j_data)} nodes for Neo4j import")
        
        return results


if __name__ == "__main__":
    # Run example
    asyncio.run(example_api_usage()) 