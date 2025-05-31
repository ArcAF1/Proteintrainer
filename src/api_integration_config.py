"""
Biomedical API Integration Configuration
Priority APIs for creatine, workout science, and health optimization
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    """Configuration for a biomedical API."""
    name: str
    base_url: str
    rate_limit: float  # requests per second
    requires_key: bool
    priority: int  # 1-5, higher is more important
    data_types: List[str]
    example_endpoints: Dict[str, str]


# Priority APIs for your use case
API_CONFIGS = {
    "pubmed": APIConfig(
        name="NCBI E-utilities (PubMed)",
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        rate_limit=3.0,  # 3/sec without key, 10/sec with free key
        requires_key=False,
        priority=5,
        data_types=["abstracts", "metadata", "citations"],
        example_endpoints={
            "search": "esearch.fcgi?db=pubmed&term=creatine+muscle&retmode=json",
            "fetch": "efetch.fcgi?db=pubmed&id={pmid}&retmode=xml",
            "summary": "esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
        }
    ),
    
    "europe_pmc": APIConfig(
        name="Europe PMC",
        base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/",
        rate_limit=10.0,  # No official limit, but be respectful
        requires_key=False,
        priority=5,
        data_types=["full_text", "abstracts", "references"],
        example_endpoints={
            "search": "search?query=creatine&format=json",
            "article": "search/PMC/{pmcid}/fullTextXML",
            "citations": "{pmid}/citations?format=json"
        }
    ),
    
    "clinicaltrials": APIConfig(
        name="ClinicalTrials.gov",
        base_url="https://clinicaltrials.gov/api/v2/",
        rate_limit=5.0,
        requires_key=False,
        priority=5,
        data_types=["trials", "interventions", "outcomes"],
        example_endpoints={
            "search": "studies?query.term=creatine&format=json",
            "study": "studies/{nct_id}",
            "fields": "studies/metadata"
        }
    ),
    
    "pubchem": APIConfig(
        name="PubChem PUG REST",
        base_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/",
        rate_limit=5.0,
        requires_key=False,
        priority=4,
        data_types=["compounds", "bioassays", "pathways"],
        example_endpoints={
            "compound": "compound/name/creatine/JSON",
            "synonyms": "compound/cid/586/synonyms/JSON",
            "bioactivity": "compound/cid/586/assaysummary/JSON"
        }
    ),
    
    "chebi": APIConfig(
        name="ChEBI",
        base_url="https://www.ebi.ac.uk/ols/api/",
        rate_limit=10.0,
        requires_key=False,
        priority=4,
        data_types=["ontology", "chemical_structure", "relationships"],
        example_endpoints={
            "search": "search?q=creatine&ontology=chebi",
            "term": "ontologies/chebi/terms?iri=http://purl.obolibrary.org/obo/CHEBI_16919",
            "hierarchy": "ontologies/chebi/hierarchicalDescendants?id=CHEBI:16919"
        }
    ),
    
    "mesh": APIConfig(
        name="MeSH",
        base_url="https://id.nlm.nih.gov/mesh/",
        rate_limit=10.0,
        requires_key=False,
        priority=5,
        data_types=["ontology", "hierarchy", "mappings"],
        example_endpoints={
            "lookup": "lookup/descriptor?label=Creatine&match=exact&limit=10",
            "sparql": "sparql?query=SELECT+*+WHERE+{?s+rdfs:label+'Creatine'@en}",
            "rdf": "2024/D003401.nt"  # Direct RDF for creatine
        }
    ),
    
    "openalex": APIConfig(
        name="OpenAlex",
        base_url="https://api.openalex.org/",
        rate_limit=100.0,  # 100k/day = ~1.15/sec sustained
        requires_key=False,
        priority=4,
        data_types=["papers", "authors", "concepts", "citations"],
        example_endpoints={
            "works": "works?filter=concepts.id:C2776834827",  # Exercise science concept
            "search": "works?search=creatine+supplementation",
            "concept": "concepts/C2776834827"  # Get concept details
        }
    ),
    
    "supp_ai": APIConfig(
        name="SUPP.AI",
        base_url="https://supp.ai/api/v1/",
        rate_limit=5.0,
        requires_key=False,
        priority=3,
        data_types=["interactions", "evidence", "supplements"],
        example_endpoints={
            "search": "agents?query=creatine",
            "interactions": "interactions?agent1=creatine",
            "evidence": "evidence?agent=creatine"
        }
    )
}


# Creatine-specific queries for each API
CREATINE_QUERIES = {
    "pubmed": [
        "creatine[MeSH Terms] AND muscle",
        "creatine supplementation AND exercise",
        "creatine monohydrate AND performance"
    ],
    "europe_pmc": [
        "creatine AND muscle AND (OPEN_ACCESS:y)",
        "creatine absorption enhancement",
        "creatine timing protocol"
    ],
    "clinicaltrials": [
        "creatine AND muscle",
        "creatine monohydrate",
        "creatine AND exercise performance"
    ],
    "mesh": [
        "D003401",  # Creatine MeSH ID
        "D018482",  # Muscle Strength MeSH ID
        "D064797"   # Physical Performance MeSH ID
    ]
}


# Integration priority for M1 MacBook (considering performance)
INTEGRATION_PHASES = {
    "phase1": {
        "apis": ["mesh", "chebi", "pubchem"],  # Ontologies first
        "description": "Build foundational ontology layer",
        "estimated_nodes": 50000,
        "estimated_time": "10-15 minutes"
    },
    "phase2": {
        "apis": ["pubmed", "europe_pmc"],  # Literature
        "description": "Import key literature metadata",
        "estimated_nodes": 100000,
        "estimated_time": "30-45 minutes"
    },
    "phase3": {
        "apis": ["clinicaltrials", "supp_ai"],  # Evidence
        "description": "Add clinical evidence and interactions",
        "estimated_nodes": 25000,
        "estimated_time": "15-20 minutes"
    },
    "phase4": {
        "apis": ["openalex"],  # Citations and concepts
        "description": "Enrich with citation networks",
        "estimated_nodes": 50000,
        "estimated_time": "20-30 minutes"
    }
} 