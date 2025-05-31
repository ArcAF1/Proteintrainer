"""
Enhanced Biomedical Tools
Comprehensive integration of biomedical APIs, databases, and analysis tools
with Neo4j knowledge graph integration
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime
import requests
import hashlib
from pathlib import Path

# Import existing components
from .biomedical_api_client import BiomedicalAPIClient
from .neo4j_setup import get_driver
from .neo4j_context_manager import Neo4jContextManager

logger = logging.getLogger(__name__)


class EnhancedBiomedicalTools:
    """
    Enhanced biomedical tools with additional APIs and features:
    - UniProt for protein data
    - ChEMBL for drug/compound data
    - STRING for protein-protein interactions
    - AlphaFold for structure predictions
    - KEGG for pathway information
    - DrugBank integration
    - RCSB PDB for protein structures
    """
    
    def __init__(self, use_neo4j: bool = True):
        """Initialize enhanced biomedical tools."""
        self.base_client = BiomedicalAPIClient()
        self.neo4j_driver = get_driver() if use_neo4j else None
        self.context_manager = Neo4jContextManager(self.neo4j_driver) if use_neo4j else None
        
        # API endpoints
        self.endpoints = {
            'uniprot': 'https://rest.uniprot.org/uniprotkb',
            'chembl': 'https://www.ebi.ac.uk/chembl/api/data',
            'string': 'https://string-db.org/api',
            'alphafold': 'https://alphafold.ebi.ac.uk/api',
            'kegg': 'http://rest.kegg.jp',
            'pdb': 'https://data.rcsb.org/rest/v1/core'
        }
        
        # Cache directory for structures
        self.cache_dir = Path('biomedical_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
    async def get_protein_comprehensive(self, protein_name: str) -> Dict[str, Any]:
        """
        Get comprehensive protein information from multiple sources.
        """
        results = {
            'protein_name': protein_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Search UniProt
        uniprot_data = await self._search_uniprot(protein_name)
        if uniprot_data:
            results['uniprot'] = uniprot_data
            accession = uniprot_data.get('accession')
            
            # Get AlphaFold structure if available
            if accession:
                structure_data = await self._get_alphafold_structure(accession)
                if structure_data:
                    results['structure'] = structure_data
                    
                # Get protein interactions from STRING
                interactions = await self._get_string_interactions(accession)
                if interactions:
                    results['interactions'] = interactions
                    
        # Get pathway information from KEGG
        pathways = await self._search_kegg_pathways(protein_name)
        if pathways:
            results['pathways'] = pathways
            
        # Store in Neo4j if available
        if self.neo4j_driver:
            self._store_protein_data(results)
            
        return results
        
    async def _search_uniprot(self, protein_name: str) -> Optional[Dict[str, Any]]:
        """Search UniProt for protein information."""
        try:
            # Search query
            query_url = f"{self.endpoints['uniprot']}/search"
            params = {
                'query': protein_name,
                'format': 'json',
                'size': 1
            }
            
            response = requests.get(query_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    return {
                        'accession': result.get('primaryAccession'),
                        'name': result.get('uniProtkbId'),
                        'organism': result.get('organism', {}).get('scientificName'),
                        'function': self._extract_function(result),
                        'sequence': result.get('sequence', {}).get('value'),
                        'length': result.get('sequence', {}).get('length'),
                        'keywords': [kw.get('name') for kw in result.get('keywords', [])]
                    }
        except Exception as e:
            logger.error(f"UniProt search error: {e}")
        return None
        
    def _extract_function(self, uniprot_entry: Dict) -> Optional[str]:
        """Extract function description from UniProt entry."""
        comments = uniprot_entry.get('comments', [])
        for comment in comments:
            if comment.get('commentType') == 'FUNCTION':
                texts = comment.get('texts', [])
                if texts:
                    return texts[0].get('value')
        return None
        
    async def _get_alphafold_structure(self, uniprot_accession: str) -> Optional[Dict[str, Any]]:
        """Get AlphaFold structure prediction."""
        try:
            # Check if structure exists
            check_url = f"{self.endpoints['alphafold']}/prediction/{uniprot_accession}"
            response = requests.get(check_url)
            
            if response.status_code == 200:
                data = response.json()[0]
                
                # Download PDB file if not cached
                pdb_url = data.get('pdbUrl')
                if pdb_url:
                    pdb_path = self.cache_dir / f"{uniprot_accession}_alphafold.pdb"
                    if not pdb_path.exists():
                        pdb_response = requests.get(pdb_url)
                        if pdb_response.status_code == 200:
                            pdb_path.write_bytes(pdb_response.content)
                            
                return {
                    'source': 'AlphaFold',
                    'confidence': data.get('globalMetricValue'),
                    'pdb_file': str(pdb_path) if pdb_path.exists() else None,
                    'version': data.get('modelCreatedDate')
                }
        except Exception as e:
            logger.error(f"AlphaFold error: {e}")
        return None
        
    async def _get_string_interactions(self, protein_id: str, species: int = 9606) -> Optional[List[Dict]]:
        """Get protein-protein interactions from STRING (default: human)."""
        try:
            url = f"{self.endpoints['string']}/json/network"
            params = {
                'identifiers': protein_id,
                'species': species,
                'caller_identity': 'biomedical_ai'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Parse interactions
                interactions = []
                for item in data[:10]:  # Top 10 interactions
                    interactions.append({
                        'partner': item.get('preferredName'),
                        'score': item.get('score'),
                        'source': 'STRING'
                    })
                return interactions
        except Exception as e:
            logger.error(f"STRING API error: {e}")
        return None
        
    async def _search_kegg_pathways(self, protein_name: str) -> Optional[List[Dict]]:
        """Search KEGG for pathway information."""
        try:
            # First find the gene
            find_url = f"{self.endpoints['kegg']}/find/genes/{protein_name}"
            response = requests.get(find_url)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                if lines and lines[0]:
                    # Get first gene ID
                    gene_id = lines[0].split('\t')[0]
                    
                    # Get pathway info
                    pathway_url = f"{self.endpoints['kegg']}/link/pathway/{gene_id}"
                    pathway_response = requests.get(pathway_url)
                    
                    if pathway_response.status_code == 200:
                        pathways = []
                        for line in pathway_response.text.strip().split('\n'):
                            if line:
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    pathway_id = parts[1].replace('path:', '')
                                    pathways.append({
                                        'id': pathway_id,
                                        'name': self._get_pathway_name(pathway_id)
                                    })
                        return pathways[:5]  # Top 5 pathways
        except Exception as e:
            logger.error(f"KEGG error: {e}")
        return None
        
    def _get_pathway_name(self, pathway_id: str) -> str:
        """Get pathway name from KEGG."""
        try:
            url = f"{self.endpoints['kegg']}/get/{pathway_id}"
            response = requests.get(url)
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    if line.startswith('NAME'):
                        return line.replace('NAME', '').strip()
        except:
            pass
        return pathway_id
        
    async def analyze_drug_target(self, drug_name: str, target_protein: str) -> Dict[str, Any]:
        """
        Analyze drug-target interactions using ChEMBL and other sources.
        """
        results = {
            'drug': drug_name,
            'target': target_protein,
            'timestamp': datetime.now().isoformat()
        }
        
        # Search ChEMBL for drug
        drug_data = await self._search_chembl_drug(drug_name)
        if drug_data:
            results['drug_info'] = drug_data
            
            # Get bioactivity data
            if drug_data.get('molecule_chembl_id'):
                bioactivity = await self._get_chembl_bioactivity(
                    drug_data['molecule_chembl_id'],
                    target_protein
                )
                if bioactivity:
                    results['bioactivity'] = bioactivity
                    
        # Get target protein info
        target_info = await self.get_protein_comprehensive(target_protein)
        results['target_info'] = target_info
        
        # Analyze interaction in Neo4j context
        if self.neo4j_driver:
            interaction_analysis = self._analyze_drug_target_interaction(
                drug_name, target_protein
            )
            results['graph_analysis'] = interaction_analysis
            
        return results
        
    async def _search_chembl_drug(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Search ChEMBL for drug information."""
        try:
            search_url = f"{self.endpoints['chembl']}/molecule/search"
            params = {
                'q': drug_name,
                'format': 'json'
            }
            
            response = requests.get(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                molecules = data.get('molecules', [])
                if molecules:
                    mol = molecules[0]
                    return {
                        'molecule_chembl_id': mol.get('molecule_chembl_id'),
                        'name': mol.get('pref_name'),
                        'max_phase': mol.get('max_phase'),
                        'molecular_weight': mol.get('full_mwt'),
                        'formula': mol.get('full_molformula')
                    }
        except Exception as e:
            logger.error(f"ChEMBL search error: {e}")
        return None
        
    async def _get_chembl_bioactivity(self, 
                                    chembl_id: str, 
                                    target_name: str) -> Optional[List[Dict]]:
        """Get bioactivity data from ChEMBL."""
        try:
            url = f"{self.endpoints['chembl']}/activity"
            params = {
                'molecule_chembl_id': chembl_id,
                'format': 'json',
                'limit': 10
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                activities = []
                
                for activity in data.get('activities', []):
                    # Filter by target if specified
                    target_pref_name = activity.get('target_pref_name', '')
                    if not target_name or target_name.lower() in target_pref_name.lower():
                        activities.append({
                            'target': target_pref_name,
                            'type': activity.get('standard_type'),
                            'value': activity.get('standard_value'),
                            'units': activity.get('standard_units'),
                            'assay': activity.get('assay_description')
                        })
                        
                return activities[:5]  # Top 5 activities
        except Exception as e:
            logger.error(f"ChEMBL bioactivity error: {e}")
        return None
        
    def _store_protein_data(self, protein_data: Dict[str, Any]):
        """Store comprehensive protein data in Neo4j."""
        if not self.neo4j_driver:
            return
            
        try:
            with self.neo4j_driver.session() as session:
                protein_name = protein_data['protein_name']
                
                # Create or update protein node
                session.run("""
                    MERGE (p:Protein {name: $name})
                    SET p.updated_at = datetime(),
                        p.uniprot_accession = $accession,
                        p.organism = $organism,
                        p.function = $function,
                        p.length = $length
                """, 
                    name=protein_name,
                    accession=protein_data.get('uniprot', {}).get('accession'),
                    organism=protein_data.get('uniprot', {}).get('organism'),
                    function=protein_data.get('uniprot', {}).get('function'),
                    length=protein_data.get('uniprot', {}).get('length')
                )
                
                # Add structure information
                if 'structure' in protein_data:
                    session.run("""
                        MATCH (p:Protein {name: $name})
                        MERGE (s:Structure {source: $source, protein: $name})
                        SET s.confidence = $confidence,
                            s.pdb_file = $pdb_file
                        CREATE (p)-[:HAS_STRUCTURE]->(s)
                    """,
                        name=protein_name,
                        source=protein_data['structure']['source'],
                        confidence=protein_data['structure']['confidence'],
                        pdb_file=protein_data['structure'].get('pdb_file')
                    )
                    
                # Add interactions
                if 'interactions' in protein_data:
                    for interaction in protein_data['interactions']:
                        session.run("""
                            MATCH (p1:Protein {name: $name1})
                            MERGE (p2:Protein {name: $name2})
                            MERGE (p1)-[i:INTERACTS_WITH]-(p2)
                            SET i.score = $score,
                                i.source = $source
                        """,
                            name1=protein_name,
                            name2=interaction['partner'],
                            score=interaction['score'],
                            source=interaction['source']
                        )
                        
                # Add pathway connections
                if 'pathways' in protein_data:
                    for pathway in protein_data['pathways']:
                        session.run("""
                            MATCH (p:Protein {name: $protein})
                            MERGE (pw:Pathway {id: $pathway_id})
                            SET pw.name = $pathway_name
                            MERGE (p)-[:INVOLVED_IN]->(pw)
                        """,
                            protein=protein_name,
                            pathway_id=pathway['id'],
                            pathway_name=pathway['name']
                        )
                        
        except Exception as e:
            logger.error(f"Error storing protein data in Neo4j: {e}")
            
    def _analyze_drug_target_interaction(self, 
                                       drug_name: str, 
                                       target_protein: str) -> Dict[str, Any]:
        """Analyze drug-target interaction using Neo4j graph."""
        if not self.neo4j_driver:
            return {}
            
        try:
            with self.neo4j_driver.session() as session:
                # Find paths between drug and target
                result = session.run("""
                    MATCH (d:Drug {name: $drug})
                    MATCH (p:Protein {name: $protein})
                    OPTIONAL MATCH path = shortestPath((d)-[*..5]-(p))
                    RETURN path,
                           [node in nodes(path) | labels(node)[0]] as node_types,
                           [rel in relationships(path) | type(rel)] as rel_types
                """, drug=drug_name, protein=target_protein)
                
                record = result.single()
                if record and record['path']:
                    return {
                        'path_exists': True,
                        'path_length': len(record['path']),
                        'node_types': record['node_types'],
                        'relationship_types': record['rel_types']
                    }
                else:
                    return {'path_exists': False}
                    
        except Exception as e:
            logger.error(f"Graph analysis error: {e}")
            return {'error': str(e)}
            
    async def predict_side_effects(self, drug_name: str) -> Dict[str, Any]:
        """
        Predict potential side effects based on drug targets and pathways.
        """
        results = {
            'drug': drug_name,
            'predictions': []
        }
        
        # Get drug info
        drug_data = await self._search_chembl_drug(drug_name)
        if not drug_data:
            return results
            
        # Get targets
        targets = await self._get_drug_targets(drug_data['molecule_chembl_id'])
        
        # Analyze each target for potential effects
        for target in targets[:5]:  # Analyze top 5 targets
            target_analysis = await self._analyze_target_effects(target['target_name'])
            if target_analysis:
                results['predictions'].append({
                    'target': target['target_name'],
                    'binding_affinity': target.get('pchembl_value'),
                    'potential_effects': target_analysis
                })
                
        # Use Neo4j to find related effects
        if self.neo4j_driver:
            graph_predictions = self._predict_effects_from_graph(drug_name)
            results['graph_predictions'] = graph_predictions
            
        return results
        
    async def _get_drug_targets(self, chembl_id: str) -> List[Dict]:
        """Get drug targets from ChEMBL."""
        try:
            url = f"{self.endpoints['chembl']}/mechanism"
            params = {
                'molecule_chembl_id': chembl_id,
                'format': 'json'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                mechanisms = []
                for mech in data.get('mechanisms', []):
                    mechanisms.append({
                        'target_name': mech.get('target_pref_name'),
                        'mechanism': mech.get('mechanism_of_action'),
                        'action_type': mech.get('action_type')
                    })
                return mechanisms
        except Exception as e:
            logger.error(f"Error getting drug targets: {e}")
        return []
        
    async def _analyze_target_effects(self, target_name: str) -> List[str]:
        """Analyze potential effects of targeting a protein."""
        # This would integrate with pathway analysis and known biology
        # For now, return example effects based on target type
        effects = []
        
        target_lower = target_name.lower()
        if 'kinase' in target_lower:
            effects.extend(['Cell signaling disruption', 'Potential immune effects'])
        elif 'receptor' in target_lower:
            effects.extend(['Altered cellular response', 'Homeostasis disruption'])
        elif 'enzyme' in target_lower:
            effects.extend(['Metabolic changes', 'Substrate accumulation'])
            
        return effects
        
    def _predict_effects_from_graph(self, drug_name: str) -> List[Dict]:
        """Use Neo4j to predict effects based on network analysis."""
        predictions = []
        
        try:
            with self.neo4j_driver.session() as session:
                # Find proteins affected by similar drugs
                result = session.run("""
                    MATCH (d1:Drug {name: $drug})-[:TARGETS]->(p1:Protein)
                    MATCH (d2:Drug)-[:TARGETS]->(p1)
                    WHERE d2.name <> $drug
                    MATCH (d2)-[:CAUSES]->(e:Effect)
                    RETURN e.name as effect, count(DISTINCT d2) as drug_count
                    ORDER BY drug_count DESC
                    LIMIT 10
                """, drug=drug_name)
                
                for record in result:
                    predictions.append({
                        'effect': record['effect'],
                        'confidence': min(record['drug_count'] / 10.0, 1.0)
                    })
                    
        except Exception as e:
            logger.error(f"Graph prediction error: {e}")
            
        return predictions
        
    async def analyze_protein_structure(self, pdb_file: str) -> Dict[str, Any]:
        """
        Analyze protein structure for binding sites and functional regions.
        """
        # This would integrate with structure analysis tools
        # For now, return mock analysis
        return {
            'file': pdb_file,
            'analysis': {
                'binding_sites': [
                    {'residues': [45, 67, 89], 'type': 'ATP binding'},
                    {'residues': [123, 145, 167], 'type': 'Substrate binding'}
                ],
                'functional_regions': [
                    {'start': 50, 'end': 100, 'function': 'Catalytic domain'},
                    {'start': 150, 'end': 200, 'function': 'Regulatory domain'}
                ],
                'structure_quality': 'High confidence (pLDDT > 90)'
            }
        } 