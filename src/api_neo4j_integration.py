"""
API-Neo4j Integration - Import biomedical API data into knowledge graph
Designed for incremental updates without overloading M1 MacBook
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from neo4j import AsyncGraphDatabase
from .biomedical_api_client import BiomedicalAPIClient
from .api_integration_config import CREATINE_QUERIES, INTEGRATION_PHASES

logger = logging.getLogger(__name__)


class APIGraphIntegrator:
    """Integrates API data into Neo4j knowledge graph."""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        self.api_client = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self.api_client = BiomedicalAPIClient()
        await self.api_client.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.driver:
            await self.driver.close()
        if self.api_client:
            await self.api_client.__aexit__(exc_type, exc_val, exc_tb)
            
    async def create_indexes(self):
        """Create indexes for efficient querying."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.pmid)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.pmcid)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.doi)",
            "CREATE INDEX IF NOT EXISTS FOR (t:ClinicalTrial) ON (t.nct_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Compound) ON (c.name)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Compound) ON (c.pubchem_cid)",
            "CREATE INDEX IF NOT EXISTS FOR (m:MeshTerm) ON (m.mesh_id)",
            "CREATE INDEX IF NOT EXISTS FOR (m:MeshTerm) ON (m.name)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Substance) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (o:Outcome) ON (o.name)"
        ]
        
        async with self.driver.session() as session:
            for index_query in indexes:
                await session.run(index_query)
                logger.info(f"Created index: {index_query.split('ON')[1].strip()}")
                
    async def import_mesh_ontology(self):
        """Import MeSH ontology for creatine and related terms."""
        logger.info("Importing MeSH ontology...")
        
        # Key MeSH terms for creatine research
        mesh_terms = [
            "Creatine",
            "Muscle Strength",
            "Physical Performance",
            "Exercise",
            "Dietary Supplements",
            "Athletic Performance"
        ]
        
        async with self.driver.session() as session:
            for term in mesh_terms:
                mesh_data = await self.api_client.search_mesh(term)
                
                # Create MeSH term nodes
                for item in mesh_data:
                    await session.run("""
                        MERGE (m:MeshTerm {mesh_id: $mesh_id})
                        SET m.name = $name,
                            m.tree_numbers = $tree_numbers,
                            m.updated = datetime()
                    """, {
                        "mesh_id": item.get("resource", "").split("/")[-1],
                        "name": item.get("label"),
                        "tree_numbers": item.get("treeNumber", [])
                    })
                    
        logger.info("MeSH ontology import complete")
        
    async def import_compound_data(self, compound_name: str = "creatine"):
        """Import compound data from PubChem."""
        logger.info(f"Importing compound data for {compound_name}...")
        
        compound_info = await self.api_client.get_compound_info(compound_name)
        
        if not compound_info:
            logger.warning(f"No compound data found for {compound_name}")
            return
            
        async with self.driver.session() as session:
            # Extract compound properties
            cid = compound_info.get("id", {}).get("id", {}).get("cid")
            
            await session.run("""
                MERGE (c:Compound {pubchem_cid: $cid})
                SET c.name = $name,
                    c.molecular_formula = $formula,
                    c.molecular_weight = $weight,
                    c.iupac_name = $iupac,
                    c.updated = datetime()
                    
                WITH c
                MERGE (s:Substance {name: $name})
                MERGE (s)-[:IS_COMPOUND]->(c)
                
                WITH s
                MATCH (m:MeshTerm {name: $name})
                MERGE (s)-[:HAS_MESH_TERM]->(m)
            """, {
                "cid": cid,
                "name": compound_name.title(),
                "formula": compound_info.get("props", [{}])[0].get("value", {}).get("sval", ""),
                "weight": compound_info.get("props", [{}])[1].get("value", {}).get("fval", 0),
                "iupac": compound_info.get("props", [{}])[2].get("value", {}).get("sval", "")
            })
            
        logger.info(f"Compound data imported for {compound_name}")
        
    async def import_publications(self, query: str, max_results: int = 100):
        """Import publications from PubMed and Europe PMC."""
        logger.info(f"Importing publications for query: {query}")
        
        # Search both sources
        results = await self.api_client.search_all_sources(
            query, 
            sources=["pubmed", "europe_pmc"],
            max_results_per_source=max_results
        )
        
        async with self.driver.session() as session:
            # Import PubMed articles (just IDs for now)
            for pmid in results.get("pubmed", []):
                await session.run("""
                    MERGE (p:Publication {pmid: $pmid})
                    SET p.source = 'PubMed',
                        p.updated = datetime()
                """, {"pmid": pmid})
                
            # Import Europe PMC articles (with metadata)
            for article in results.get("europe_pmc", []):
                await session.run("""
                    MERGE (p:Publication {pmid: $pmid})
                    SET p.pmcid = $pmcid,
                        p.title = $title,
                        p.abstract = $abstract,
                        p.doi = $doi,
                        p.source = 'EuropePMC',
                        p.updated = datetime()
                """, {
                    "pmid": article.get("pmid", f"PMC{article.get('pmcid', 'unknown')}"),
                    "pmcid": article.get("pmcid"),
                    "title": article.get("title", ""),
                    "abstract": article.get("abstractText", ""),
                    "doi": article.get("doi", "")
                })
                
        logger.info(f"Imported {len(results.get('pubmed', []))} PubMed articles")
        logger.info(f"Imported {len(results.get('europe_pmc', []))} Europe PMC articles")
        
    async def import_clinical_trials(self, query: str, max_results: int = 50):
        """Import clinical trials data."""
        logger.info(f"Importing clinical trials for query: {query}")
        
        trials = await self.api_client.search_clinical_trials(query, max_results)
        
        async with self.driver.session() as session:
            for trial in trials:
                protocol = trial.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                design_module = protocol.get("designModule", {})
                conditions_module = protocol.get("conditionsModule", {})
                interventions_module = protocol.get("armsInterventionsModule", {})
                
                nct_id = id_module.get("nctId")
                if not nct_id:
                    continue
                    
                # Create trial node
                await session.run("""
                    MERGE (t:ClinicalTrial {nct_id: $nct_id})
                    SET t.title = $title,
                        t.status = $status,
                        t.phase = $phase,
                        t.updated = datetime()
                """, {
                    "nct_id": nct_id,
                    "title": id_module.get("briefTitle", ""),
                    "status": protocol.get("statusModule", {}).get("overallStatus", ""),
                    "phase": design_module.get("phases", ["Unknown"])[0]
                })
                
                # Link to conditions
                for condition in conditions_module.get("conditions", []):
                    await session.run("""
                        MATCH (t:ClinicalTrial {nct_id: $nct_id})
                        MERGE (c:Condition {name: $condition})
                        MERGE (t)-[:STUDIES_CONDITION]->(c)
                    """, {"nct_id": nct_id, "condition": condition})
                    
                # Link to interventions
                for intervention in interventions_module.get("interventions", []):
                    intervention_name = intervention.get("name", "")
                    intervention_type = intervention.get("type", "")
                    
                    if intervention_name:
                        await session.run("""
                            MATCH (t:ClinicalTrial {nct_id: $nct_id})
                            MERGE (i:Intervention {name: $name})
                            SET i.type = $type
                            MERGE (t)-[:USES_INTERVENTION]->(i)
                            
                            WITH i
                            WHERE toLower(i.name) CONTAINS 'creatine'
                            MATCH (s:Substance {name: 'Creatine'})
                            MERGE (i)-[:INVOLVES_SUBSTANCE]->(s)
                        """, {
                            "nct_id": nct_id,
                            "name": intervention_name,
                            "type": intervention_type
                        })
                        
        logger.info(f"Imported {len(trials)} clinical trials")
        
    async def create_relationships(self):
        """Create relationships between entities based on co-occurrence."""
        logger.info("Creating entity relationships...")
        
        async with self.driver.session() as session:
            # Link publications to substances mentioned in title/abstract
            await session.run("""
                MATCH (p:Publication)
                WHERE p.title IS NOT NULL OR p.abstract IS NOT NULL
                WITH p, toLower(coalesce(p.title, '') + ' ' + coalesce(p.abstract, '')) AS text
                
                MATCH (s:Substance)
                WHERE text CONTAINS toLower(s.name)
                MERGE (p)-[:MENTIONS_SUBSTANCE]->(s)
            """)
            
            # Link trials to publications (by title similarity)
            await session.run("""
                MATCH (t:ClinicalTrial)
                WHERE t.title IS NOT NULL
                WITH t, toLower(t.title) AS trial_title
                
                MATCH (p:Publication)
                WHERE p.title IS NOT NULL 
                AND toLower(p.title) CONTAINS substring(trial_title, 0, 20)
                MERGE (t)-[:HAS_PUBLICATION]->(p)
            """)
            
            # Create outcome relationships
            await session.run("""
                // Create common outcomes
                FOREACH (outcome IN ['Muscle Strength', 'Muscle Mass', 'Exercise Performance', 
                                    'Fatigue', 'Recovery', 'Power Output'] |
                    MERGE (:Outcome {name: outcome})
                )
                
                // Link substances to outcomes based on trial evidence
                MATCH (t:ClinicalTrial)-[:USES_INTERVENTION]->(i:Intervention)-[:INVOLVES_SUBSTANCE]->(s:Substance)
                WHERE t.title CONTAINS 'muscle' OR t.title CONTAINS 'strength' OR t.title CONTAINS 'performance'
                MATCH (o:Outcome)
                WHERE toLower(t.title) CONTAINS toLower(o.name)
                MERGE (s)-[:IMPROVES {source: t.nct_id}]->(o)
            """)
            
        logger.info("Relationships created")
        
    async def import_phase(self, phase_name: str):
        """Import data for a specific phase."""
        phase = INTEGRATION_PHASES.get(phase_name)
        if not phase:
            logger.error(f"Unknown phase: {phase_name}")
            return
            
        logger.info(f"Starting {phase_name}: {phase['description']}")
        start_time = datetime.now()
        
        for api in phase["apis"]:
            if api == "mesh":
                await self.import_mesh_ontology()
            elif api == "pubchem":
                await self.import_compound_data("creatine")
            elif api in ["pubmed", "europe_pmc"]:
                for query in CREATINE_QUERIES.get(api, [])[:2]:  # Limit queries
                    await self.import_publications(query, max_results=50)
            elif api == "clinicaltrials":
                for query in CREATINE_QUERIES.get(api, [])[:2]:
                    await self.import_clinical_trials(query, max_results=25)
                    
        # Create relationships after importing data
        await self.create_relationships()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Phase {phase_name} complete in {elapsed:.1f} seconds")
        
    async def run_full_import(self):
        """Run full import in phases."""
        await self.create_indexes()
        
        for phase_name in ["phase1", "phase2", "phase3"]:
            await self.import_phase(phase_name)
            
            # Brief pause between phases
            await asyncio.sleep(5)
            
        logger.info("Full import complete!")
        
    async def get_import_stats(self) -> Dict[str, int]:
        """Get statistics about imported data."""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)
            
            stats = {}
            async for record in result:
                stats[record["label"]] = record["count"]
                
            # Get relationship counts
            rel_result = await session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            
            stats["relationships"] = {}
            async for record in rel_result:
                stats["relationships"][record["type"]] = record["count"]
                
        return stats


# Example usage
async def example_integration():
    """Example of how to use the API-Neo4j integration."""
    async with APIGraphIntegrator() as integrator:
        # Import just phase 1 (ontologies)
        await integrator.import_phase("phase1")
        
        # Get stats
        stats = await integrator.get_import_stats()
        print("\nImport Statistics:")
        for label, count in stats.items():
            if label != "relationships":
                print(f"  {label}: {count}")
                
        if "relationships" in stats:
            print("\nRelationships:")
            for rel_type, count in stats["relationships"].items():
                print(f"  {rel_type}: {count}")


if __name__ == "__main__":
    asyncio.run(example_integration()) 