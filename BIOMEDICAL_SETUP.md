# Advanced Biomedical Package Setup

This guide covers installation of specialized biomedical packages that require additional steps beyond `pip install`.

## RDKit (Chemical Informatics)

RDKit is a powerful cheminformatics toolkit for drug discovery and chemical analysis.

### Installation Options

#### Option 1: Conda (Recommended)
```bash
# In your virtual environment
conda install -c conda-forge rdkit
```

#### Option 2: pip (Simple but may have limited features)
```bash
pip install rdkit
```

#### Option 3: Build from Source (Advanced)
See: https://rdkit.org/docs/Install.html

### Verification
```python
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')
print(f"Molecule has {mol.GetNumAtoms()} atoms")
```

## Neo4j GraphRAG

Already included in requirements.txt as `neo4j-graphrag>=1.7.0`.

### Optional Extensions
```bash
# For specific LLM providers
pip install "neo4j-graphrag[openai]"      # OpenAI support
pip install "neo4j-graphrag[ollama]"      # Ollama support  
pip install "neo4j-graphrag[anthropic]"   # Anthropic support
pip install "neo4j-graphrag[experimental]" # Experimental features
```

## Bio-embeddings (Advanced Protein Embeddings)

Specialized embeddings for protein sequences.

### Installation
```bash
# Install from conda-forge (recommended)
conda install -c conda-forge bio-embeddings

# Or build from source
pip install git+https://github.com/sacdallago/bio_embeddings.git
```

## Additional Medical NLP Tools

### ScispaCy Models
```bash
# Install specific medical models after scispacy is installed
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

### BioGPT Models
```bash
# For biomedical text generation
pip install transformers[bio]
```

## Medical Datasets

### MIMIC-III/IV Setup
```python
# Requires credentialed access to PhysioNet
# See: https://mimic.mit.edu/docs/gettingstarted/
```

### PubMed Tools
```bash
# Enhanced PubMed access
pip install biopython pubchempy
pip install pymed  # PubMed API wrapper
```

## Chemical Database Integration

### ChEMBL Access
```python
# Already available through requests, but for enhanced access:
pip install chembl-webresource-client
```

### PubChem Integration
```python
# Already included as pubchempy in requirements.txt
from pubchempy import Compound
```

## GPU Acceleration for M1 Macs

### Apple Metal Performance Shaders
```bash
# Already included in requirements.txt as mlx
# For additional acceleration:
pip install tensorflow-metal
```

## Verification Script

Create `test_biomedical_setup.py`:

```python
#!/usr/bin/env python3
"""Test biomedical package installations."""

def test_rdkit():
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CCO')
        print(f"✅ RDKit: Ethanol has {mol.GetNumAtoms()} atoms")
        return True
    except ImportError:
        print("❌ RDKit not installed")
        return False

def test_neo4j_graphrag():
    try:
        from neo4j_graphrag.embeddings import OpenAIEmbeddings
        print("✅ Neo4j GraphRAG: Available")
        return True
    except ImportError:
        print("❌ Neo4j GraphRAG not installed")
        return False

def test_biopython():
    try:
        from Bio import SeqIO
        print("✅ BioPython: Available")
        return True
    except ImportError:
        print("❌ BioPython not installed")
        return False

def test_scispacy():
    try:
        import scispacy
        import spacy
        print("✅ SciSpaCy: Available")
        return True
    except ImportError:
        print("❌ SciSpaCy not installed")
        return False

if __name__ == "__main__":
    print("Testing Biomedical Package Setup...")
    tests = [test_rdkit, test_neo4j_graphrag, test_biopython, test_scispacy]
    passed = sum(test() for test in tests)
    print(f"\nResults: {passed}/{len(tests)} packages working")
```

Run with: `python test_biomedical_setup.py`

## Troubleshooting

### RDKit Installation Issues
- On M1 Macs, prefer conda installation
- If pip fails, try: `pip install rdkit-pypi` (alternative package)

### Neo4j Connection Issues  
- Ensure Neo4j database is running: `docker compose up -d`
- Check connection in `src/diagnostics.py`

### Memory Issues
- RDKit and large models can use significant RAM
- Consider using smaller embedding models for development
- Use `GPUtil` to monitor GPU memory

### Package Conflicts
- Use virtual environments to isolate dependencies
- For conda+pip mixing, install conda packages first

## Integration with Main System

Once packages are installed, they're automatically detected by:
- `src/diagnostics.py` - System health checks
- `src/embeddings.py` - Enhanced embedding models  
- `src/unified_system.py` - Full system integration

The system gracefully handles missing optional packages and provides recommendations in diagnostics. 