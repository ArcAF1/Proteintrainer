"""
Biomedical Training Data Generator for LLM Fine-tuning

Converts datasets into instruction-following format:
- Hetionet → Q&A pairs (drug-disease, gene-pathway, etc.)
- ChEMBL → Molecular property predictions  
- PubMed → Medical reasoning chains
- ClinicalTrials → Treatment outcomes
- DrugBank → Drug interaction warnings
"""
from __future__ import annotations

import json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
import random
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

from .config import settings


@dataclass
class TrainingExample:
    """Single training example in instruction format."""
    instruction: str
    input: str = ""
    output: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BiomedicalDataGenerator:
    """Generate training data from biomedical datasets."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (settings.data_dir / "training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples: List[TrainingExample] = []
        
    def generate_all(self, max_examples: int = 50000) -> List[TrainingExample]:
        """Generate training data from all available sources."""
        print("🔄 Generating training data from biomedical sources...")
        
        # Try each data source
        try:
            self.generate_from_hetionet()
        except Exception as e:
            print(f"⚠️ Hetionet conversion failed: {e}")
            
        try:
            self.generate_from_chembl()
        except Exception as e:
            print(f"⚠️ ChEMBL conversion failed: {e}")
            
        try:
            self.generate_from_clinical_trials()
        except Exception as e:
            print(f"⚠️ Clinical trials conversion failed: {e}")
            
        try:
            self.generate_from_pubmed()
        except Exception as e:
            print(f"⚠️ PubMed conversion failed: {e}")
        
        # Shuffle and limit
        random.shuffle(self.examples)
        if len(self.examples) > max_examples:
            self.examples = self.examples[:max_examples]
            
        print(f"✅ Generated {len(self.examples)} training examples")
        return self.examples
    
    def generate_from_hetionet(self) -> None:
        """Convert Hetionet knowledge graph to Q&A pairs."""
        hetionet_file = settings.data_dir / "hetionet" / "hetionet-v1.0.json"
        if not hetionet_file.exists():
            # Try compressed version
            import bz2
            compressed_file = settings.data_dir / "hetionet-v1.0.json.bz2"
            if compressed_file.exists():
                with bz2.open(compressed_file, 'rt') as f:
                    data = json.load(f)
            else:
                print("No Hetionet data found")
                return
        else:
            with open(hetionet_file) as f:
                data = json.load(f)
        
        nodes = {node['identifier']: node for node in data.get('nodes', [])}
        edges = data.get('edges', [])
        
        print(f"Processing {len(edges)} Hetionet relationships...")
        
        # Generate Q&A from relationships
        for edge in tqdm(edges[:5000], desc="Hetionet edges"):  # Limit for memory
            source_id = edge.get('source')
            target_id = edge.get('target')
            rel_type = edge.get('kind')
            
            source_node = nodes.get(source_id, {})
            target_node = nodes.get(target_id, {})
            
            source_name = source_node.get('name', source_id)
            target_name = target_node.get('name', target_id)
            source_type = source_node.get('kind', 'entity')
            target_type = target_node.get('kind', 'entity')
            
            # Generate different question types
            examples = self._generate_relationship_questions(
                source_name, target_name, source_type, target_type, rel_type
            )
            
            for example in examples:
                example.metadata.update({
                    'source': 'hetionet',
                    'confidence': 0.95,
                    'relationship_type': rel_type
                })
                self.examples.append(example)
    
    def _generate_relationship_questions(self, source: str, target: str, 
                                       source_type: str, target_type: str, 
                                       rel_type: str) -> List[TrainingExample]:
        """Generate various question types from a single relationship."""
        examples = []
        
        # Drug-Disease relationships
        if source_type == "Compound" and target_type == "Disease" and "treat" in rel_type.lower():
            examples.extend([
                TrainingExample(
                    instruction=f"What diseases can {source} treat?",
                    output=f"{source} can be used to treat {target}."
                ),
                TrainingExample(
                    instruction=f"What medications are available for {target}?",
                    output=f"{source} is one medication that can be used to treat {target}."
                ),
                TrainingExample(
                    instruction="List a drug-disease treatment relationship.",
                    output=f"{source} treats {target}."
                )
            ])
            
        # Gene-Disease associations
        elif source_type == "Gene" and target_type == "Disease":
            examples.extend([
                TrainingExample(
                    instruction=f"How is the {source} gene related to {target}?",
                    output=f"The {source} gene is associated with {target}."
                ),
                TrainingExample(
                    instruction=f"What genes are involved in {target}?",
                    output=f"The {source} gene is one of the genes associated with {target}."
                )
            ])
            
        # Pathway relationships
        elif "Pathway" in source_type or "Pathway" in target_type:
            examples.append(
                TrainingExample(
                    instruction=f"Describe the relationship between {source} and {target}.",
                    output=f"{source} is connected to {target} through biological pathways."
                )
            )
            
        # Generic relationship
        if not examples:  # Fallback for any relationship type
            examples.append(
                TrainingExample(
                    instruction=f"What is the relationship between {source} and {target}?",
                    output=f"{source} is related to {target} through a {rel_type} relationship."
                )
            )
            
        return examples
    
    def generate_from_chembl(self) -> None:
        """Convert ChEMBL data to molecular property questions."""
        chembl_db = settings.data_dir / "chembl_sqlite" / "chembl_35.db"
        if not chembl_db.exists():
            print("No ChEMBL database found")
            return
            
        print("Processing ChEMBL molecular data...")
        
        conn = sqlite3.connect(chembl_db)
        
        # Query for bioactivity data
        query = """
        SELECT 
            cs.canonical_smiles,
            md.chembl_id as drug_id,
            md.pref_name as drug_name,
            act.standard_value,
            act.standard_units,
            act.standard_type,
            td.pref_name as target_name
        FROM compound_structures cs
        JOIN molecule_dictionary md ON cs.molregno = md.molregno
        JOIN activities act ON md.molregno = act.molregno
        JOIN target_dictionary td ON act.tid = td.tid
        WHERE act.standard_value IS NOT NULL 
        AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'potency')
        AND md.pref_name IS NOT NULL
        LIMIT 2000
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            print(f"Found {len(df)} ChEMBL bioactivity records")
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="ChEMBL activities"):
                examples = self._generate_chembl_questions(row)
                self.examples.extend(examples)
                
        except Exception as e:
            print(f"ChEMBL query failed: {e}")
        finally:
            conn.close()
    
    def _generate_chembl_questions(self, row) -> List[TrainingExample]:
        """Generate questions from ChEMBL bioactivity data."""
        examples = []
        
        drug_name = row['drug_name']
        target_name = row['target_name']
        value = row['standard_value']
        units = row['standard_units']
        assay_type = row['standard_type']
        
        if pd.isna(value) or pd.isna(drug_name) or pd.isna(target_name):
            return examples
            
        # Format the bioactivity value
        if units == 'nM':
            activity_desc = f"{value} nanomolar"
        elif units == 'uM':
            activity_desc = f"{value} micromolar"
        else:
            activity_desc = f"{value} {units}"
            
        examples.extend([
            TrainingExample(
                instruction=f"What is the {assay_type} of {drug_name} against {target_name}?",
                output=f"The {assay_type} of {drug_name} against {target_name} is {activity_desc}.",
                metadata={'source': 'chembl', 'confidence': 0.9, 'assay_type': assay_type}
            ),
            TrainingExample(
                instruction=f"How potent is {drug_name}?",
                output=f"{drug_name} has a {assay_type} of {activity_desc} against {target_name}.",
                metadata={'source': 'chembl', 'confidence': 0.9}
            )
        ])
        
        return examples
    
    def generate_from_clinical_trials(self) -> None:
        """Convert clinical trials data to treatment outcome questions."""
        ct_dir = settings.data_dir / "clinical_trials"
        if not ct_dir.exists():
            print("No clinical trials data found")
            return
            
        xml_files = list(ct_dir.glob("**/*.xml"))[:500]  # Limit for processing
        print(f"Processing {len(xml_files)} clinical trial files...")
        
        for xml_file in tqdm(xml_files, desc="Clinical trials"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                examples = self._parse_clinical_trial(root)
                self.examples.extend(examples)
                
            except Exception as e:
                continue  # Skip malformed XML
    
    def _parse_clinical_trial(self, root) -> List[TrainingExample]:
        """Parse a single clinical trial XML to training examples."""
        examples = []
        
        # Extract key information
        title = self._get_xml_text(root, './/brief_title')
        condition = self._get_xml_text(root, './/condition')
        intervention = self._get_xml_text(root, './/intervention_name')
        phase = self._get_xml_text(root, './/phase')
        status = self._get_xml_text(root, './/overall_status')
        
        if not all([title, condition, intervention]):
            return examples
            
        # Generate Q&A pairs
        examples.extend([
            TrainingExample(
                instruction=f"What is being studied in the clinical trial '{title}'?",
                output=f"This clinical trial is studying {intervention} for {condition}.",
                metadata={'source': 'clinical_trials', 'confidence': 0.85, 'phase': phase}
            ),
            TrainingExample(
                instruction=f"What interventions are being tested for {condition}?",
                output=f"{intervention} is being tested for {condition} in clinical trials.",
                metadata={'source': 'clinical_trials', 'confidence': 0.85}
            )
        ])
        
        if phase:
            examples.append(
                TrainingExample(
                    instruction=f"What phase is the {intervention} trial for {condition}?",
                    output=f"The {intervention} trial for {condition} is in {phase}.",
                    metadata={'source': 'clinical_trials', 'confidence': 0.85}
                )
            )
            
        return examples
    
    def _get_xml_text(self, root, xpath: str) -> Optional[str]:
        """Safely extract text from XML."""
        element = root.find(xpath)
        return element.text.strip() if element is not None and element.text else None
    
    def generate_from_pubmed(self) -> None:
        """Generate reasoning chains from PubMed abstracts."""
        pubmed_dirs = [
            settings.data_dir / "pmc_auto",
            settings.data_dir / "arxiv_auto",
        ]
        
        for pubmed_dir in pubmed_dirs:
            if not pubmed_dir.exists():
                continue
                
            txt_files = list(pubmed_dir.glob("*.txt"))[:200]  # Limit processing
            print(f"Processing {len(txt_files)} PubMed files from {pubmed_dir}")
            
            for txt_file in tqdm(txt_files, desc=f"PubMed {pubmed_dir.name}"):
                try:
                    content = txt_file.read_text(encoding='utf-8')
                    examples = self._generate_pubmed_questions(content)
                    self.examples.extend(examples)
                except Exception as e:
                    continue
    
    def _generate_pubmed_questions(self, content: str) -> List[TrainingExample]:
        """Generate questions from PubMed abstract."""
        examples = []
        
        # Extract title and abstract
        lines = content.split('\n')
        title = ""
        abstract = ""
        
        for line in lines:
            if line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
            elif line.startswith('Abstract:'):
                abstract = line.replace('Abstract:', '').strip()
        
        if not title or not abstract or len(abstract) < 100:
            return examples
            
        # Generate research questions
        examples.extend([
            TrainingExample(
                instruction="Summarize recent medical research findings.",
                output=f"Study: {title}\n\nFindings: {abstract[:300]}...",
                metadata={'source': 'pubmed', 'confidence': 0.8}
            ),
            TrainingExample(
                instruction=f"What research has been done on topics related to '{title[:50]}...'?",
                output=f"Research shows: {abstract[:200]}...",
                metadata={'source': 'pubmed', 'confidence': 0.8}
            )
        ])
        
        return examples
    
    def add_data_augmentation(self) -> None:
        """Add paraphrased and negative examples."""
        print("🔄 Adding data augmentation...")
        
        original_count = len(self.examples)
        augmented = []
        
        # Add negative examples
        for example in self.examples[:1000]:  # Limit for processing
            if random.random() < 0.1:  # 10% chance of negative example
                negative = self._create_negative_example(example)
                if negative:
                    augmented.append(negative)
        
        self.examples.extend(augmented)
        print(f"Added {len(augmented)} augmented examples ({original_count} → {len(self.examples)})")
    
    def _create_negative_example(self, example: TrainingExample) -> Optional[TrainingExample]:
        """Create a negative/safety example."""
        # Safety examples for medical content
        safety_instructions = [
            "Is it safe to self-medicate with this drug?",
            "Can I stop taking my prescribed medication?",
            "Should I ignore my doctor's advice?",
        ]
        
        safety_response = ("I cannot provide medical advice. Please consult with a qualified "
                         "healthcare professional for medical decisions. Never start, stop, or "
                         "change medications without proper medical supervision.")
        
        if random.random() < 0.5:
            return TrainingExample(
                instruction=random.choice(safety_instructions),
                output=safety_response,
                metadata={'source': 'safety', 'confidence': 1.0, 'type': 'safety'}
            )
        
        return None
    
    def save_training_data(self, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, Path]:
        """Save training data in multiple formats."""
        if not self.examples:
            raise ValueError("No training examples generated")
            
        # Split data
        random.shuffle(self.examples)
        total = len(self.examples)
        
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        
        splits = {
            'train': self.examples[:train_end],
            'validation': self.examples[train_end:val_end],
            'test': self.examples[val_end:]
        }
        
        saved_files = {}
        
        for split_name, split_data in splits.items():
            # Save as JSONL (for training)
            jsonl_file = self.output_dir / f"{split_name}.jsonl"
            with open(jsonl_file, 'w') as f:
                for example in split_data:
                    json.dump(asdict(example), f)
                    f.write('\n')
            
            # Save as Alpaca format
            alpaca_file = self.output_dir / f"{split_name}_alpaca.json"
            alpaca_data = []
            for example in split_data:
                alpaca_data.append({
                    'instruction': example.instruction,
                    'input': example.input,
                    'output': example.output
                })
            with open(alpaca_file, 'w') as f:
                json.dump(alpaca_data, f, indent=2)
            
            saved_files[split_name] = jsonl_file
            print(f"✅ Saved {len(split_data)} {split_name} examples to {jsonl_file}")
        
        return saved_files


def main():
    """Generate biomedical training data."""
    generator = BiomedicalDataGenerator()
    
    # Generate training data
    examples = generator.generate_all(max_examples=25000)
    
    # Add augmentation
    generator.add_data_augmentation()
    
    # Save training data
    files = generator.save_training_data()
    
    print(f"\n✅ Training data generation complete!")
    print(f"📊 Total examples: {len(generator.examples)}")
    print(f"📁 Files saved in: {generator.output_dir}")
    
    return files


if __name__ == "__main__":
    main() 