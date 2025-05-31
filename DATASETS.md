# Biomedical Datasets Guide

This document describes all the biomedical datasets available in Proteintrainer, their contents, sizes, and usage.

## Core Datasets

### 1. ChEMBL SQLite Database
- **Size**: 4.6GB
- **Format**: SQLite database
- **Contents**: 
  - 2.4M+ bioactive compounds
  - 1.5M+ assays
  - 15K+ targets
  - Activity data, drug mechanisms, clinical candidates
- **Use Case**: Drug discovery, target identification, bioactivity prediction
- **Update Frequency**: Quarterly

### 2. Hetionet Knowledge Graph  
- **Size**: ~50MB
- **Format**: JSON (bzip2 compressed)
- **Contents**:
  - 47K nodes (11 types: compounds, diseases, genes, pathways, etc.)
  - 2.25M relationships (24 types)
  - Integrates 29 public biomedical databases
- **Use Case**: Network medicine, drug repurposing, pathway analysis
- **License**: CC0 1.0

### 3. ClinicalTrials.gov
- **Size**: ~5GB
- **Format**: XML (zipped)
- **Contents**:
  - 450K+ clinical studies worldwide
  - Study protocols, outcomes, adverse events
  - Enrollment criteria, interventions
- **Use Case**: Clinical research, trial design, safety analysis
- **Update**: Weekly

### 4. PubMed Baseline
- **Size**: ~20GB (full set, 972+ files)
- **Format**: XML (gzipped)
- **Contents**:
  - 36M+ biomedical abstracts
  - MeSH terms, author affiliations
  - Citation networks
- **Use Case**: Literature mining, trend analysis, knowledge extraction
- **Note**: We download a subset by default

### 5. PubMed Central Open Access
- **Size**: ~100GB+ (full collection)
- **Format**: XML (tar.gz packages)
- **Contents**:
  - 3M+ full-text articles
  - Figures, tables, supplementary data
  - Commercial & non-commercial use subsets
- **Use Case**: Deep literature analysis, figure mining, methods extraction

## Chemical/Molecular Datasets

### 6. PubChem Compounds
- **Size**: ~300GB (full database)
- **Format**: SDF (gzipped)
- **Contents**:
  - 115M+ chemical structures
  - Physicochemical properties
  - Bioassay results
- **Use Case**: Chemical similarity, property prediction, virtual screening

### 7. ZINC In-Stock Compounds
- **Size**: ~1GB
- **Format**: SMILES (gzipped)
- **Contents**:
  - 14M+ purchasable compounds
  - Vendor information
  - Drug-like subset available
- **Use Case**: Virtual screening, hit-to-lead optimization

### 8. ChEMBL SDF
- **Size**: 768MB
- **Format**: SDF (gzipped)
- **Contents**:
  - Chemical structures for all ChEMBL compounds
  - Calculated properties
  - Standardized representations
- **Use Case**: Cheminformatics, molecular modeling

## Protein & Interaction Datasets

### 9. UniProt Swiss-Prot
- **Size**: ~100MB compressed
- **Format**: XML (gzipped)
- **Contents**:
  - 570K+ reviewed protein sequences
  - Functional annotations
  - Disease associations
- **Use Case**: Protein analysis, target validation

### 10. STRING Protein Links
- **Size**: ~50MB (human only)
- **Format**: TSV (gzipped)
- **Contents**:
  - Protein-protein interactions
  - Confidence scores
  - 20K+ human proteins
- **Use Case**: Network biology, pathway reconstruction

### 11. BindingDB
- **Size**: ~400MB
- **Format**: TSV (zipped)
- **Contents**:
  - 2.7M+ binding measurements
  - 1.2M+ compounds
  - 9K+ protein targets
- **Use Case**: Binding affinity prediction, SAR analysis

## Metabolomics & Pathways

### 12. Human Metabolome Database (HMDB)
- **Size**: ~2GB
- **Format**: XML/SDF (zipped)
- **Contents**:
  - 220K+ metabolite entries
  - Biological roles, pathways
  - Disease associations
- **Use Case**: Metabolomics, biomarker discovery

### 13. Reactome Pathways
- **Size**: ~5MB
- **Format**: TSV
- **Contents**:
  - 2.6K+ human pathways
  - 11K+ reactions
  - Cross-species mappings
- **Use Case**: Pathway enrichment, systems biology

## Disease & Phenotype Data

### 14. DisGeNET
- **Size**: ~10MB
- **Format**: TSV (gzipped)
- **Contents**:
  - 1.1M+ gene-disease associations
  - 30K+ genes
  - 30K+ diseases/phenotypes
- **Use Case**: Disease gene prioritization, phenotype analysis

## Usage Tips

### Storage Requirements
- **Minimal Setup**: 5-10GB (ChEMBL + Hetionet + small samples)
- **Standard Setup**: 20GB (adds clinical trials, proteins)
- **Full Setup**: 50GB+ (includes literature datasets)
- **Complete Mirror**: 500GB+ (all datasets, all files)

### Download Strategy
1. Start with Essential datasets (ChEMBL, Hetionet)
2. Add domain-specific data based on your research
3. Use API fetching for recent literature
4. Consider storage vs. compute tradeoffs

### Update Schedule
- **Daily**: PubMed updates, clinical trials
- **Weekly**: PubChem, UniProt
- **Monthly**: Most other databases
- **Quarterly**: ChEMBL, major releases

### Performance Considerations
- SQLite databases offer fastest queries
- Compressed formats save space but need decompression
- Index files after extraction for better performance
- Consider using subset for development/testing

## Data Integration

The Proteintrainer system integrates these datasets through:
1. **Knowledge Graph**: Neo4j integrates entities and relationships
2. **Vector Search**: FAISS indexes for semantic similarity
3. **Full-Text Search**: Indexed document content
4. **Cross-References**: Unified identifier mapping

## Citation Requirements

Please cite the original data sources:
- ChEMBL: Zdrazil et al., Nucleic Acids Res 2024
- Hetionet: Himmelstein et al., eLife 2017
- DisGeNET: Piñero et al., Nucleic Acids Res 2020
- (See individual dataset pages for full citations) 