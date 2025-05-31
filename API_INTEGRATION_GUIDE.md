# 🌐 Biomedical API Integration Guide

## Overview

This guide shows how to integrate open-source biomedical APIs into your local AI system to provide:
- Real-time clinical trial data
- Latest research papers
- Chemical compound information
- Standardized medical terminology (MeSH)
- Evidence-based relationships

## 🚀 Quick Start

### 1. Test API Access

```bash
# Test the API client
python -m src.biomedical_api_client
```

### 2. Import Data to Neo4j

```bash
# Run phased import (optimized for M1 MacBook)
python -c "
import asyncio
from src.api_neo4j_integration import APIGraphIntegrator

async def import_data():
    async with APIGraphIntegrator() as integrator:
        # Phase 1: Ontologies (10-15 min)
        await integrator.import_phase('phase1')
        
        # Get stats
        stats = await integrator.get_import_stats()
        print(f'Imported: {stats}')

asyncio.run(import_data())
"
```

### 3. Enable API-Enhanced RAG

```python
# In your code, use the enhanced RAG
from src.api_enhanced_rag import api_enhanced_answer

# This will combine local knowledge with live API data
response = await api_enhanced_answer("What are the latest trials on creatine?")
```

## 📊 Available APIs

### Priority APIs for Creatine Research

1. **PubMed/PMC** (Priority: ⭐⭐⭐⭐⭐)
   - 35M+ biomedical abstracts
   - No API key required (3 req/sec)
   - Perfect for literature validation

2. **ClinicalTrials.gov** (Priority: ⭐⭐⭐⭐⭐)
   - All registered clinical trials
   - Structured intervention/outcome data
   - Essential for evidence-based claims

3. **Europe PMC** (Priority: ⭐⭐⭐⭐⭐)
   - Full-text open access papers
   - More recent than PubMed alone
   - Great for latest research

4. **MeSH** (Priority: ⭐⭐⭐⭐⭐)
   - Medical terminology ontology
   - Standardizes your knowledge graph
   - Links concepts hierarchically

5. **PubChem** (Priority: ⭐⭐⭐⭐)
   - Chemical compound data
   - Molecular properties
   - Bioactivity information

## 🔧 Implementation Details

### API Client Features

- **Rate Limiting**: Respects API limits automatically
- **Caching**: 7-day cache to reduce API calls
- **Parallel Queries**: Search multiple sources simultaneously
- **Error Handling**: Graceful fallbacks if APIs fail

### Neo4j Integration

The system creates a rich knowledge graph:

```
(Substance:Creatine)-[:IMPROVES]->(Outcome:MuscleStrength)
                    \-[:IS_COMPOUND]->(Compound {formula: "C4H9N3O2"})
                    \-[:HAS_MESH_TERM]->(MeshTerm {id: "D003401"})
                    
(ClinicalTrial)-[:USES_INTERVENTION]->(Intervention)-[:INVOLVES_SUBSTANCE]->(Substance)
              \-[:STUDIES_CONDITION]->(Condition)
              \-[:HAS_PUBLICATION]->(Publication)
```

### Performance Optimization

Designed for M1 MacBook with 16GB RAM:

| Phase | APIs | Time | Memory | Nodes Added |
|-------|------|------|--------|-------------|
| 1 | MeSH, ChEBI, PubChem | 10-15 min | ~2GB | ~50k |
| 2 | PubMed, Europe PMC | 30-45 min | ~3GB | ~100k |
| 3 | ClinicalTrials, SUPP.AI | 15-20 min | ~2GB | ~25k |

## 🎯 Use Cases

### 1. Real-Time Evidence Lookup

```python
# User asks about latest creatine research
"What are the latest clinical trials on creatine for athletes?"

# System automatically:
# - Queries ClinicalTrials.gov API
# - Finds trials with status "Recruiting" or "Active"
# - Returns structured data with NCT IDs
```

### 2. Chemical Information

```python
# User asks about creatine properties
"What is the molecular structure of creatine?"

# System automatically:
# - Queries PubChem API
# - Returns molecular formula, weight, IUPAC name
# - Links to bioactivity data
```

### 3. Literature Validation

```python
# During research mode
"Research creatine absorption enhancement"

# System automatically:
# - Searches PubMed for recent papers
# - Retrieves full-text from Europe PMC
# - Validates hypotheses against literature
```

## 🔄 Update Strategy

### Daily Updates (Lightweight)
```python
# Add new papers on specific topics
await integrator.import_publications("creatine new", max_results=20)
```

### Weekly Updates (Moderate)
```python
# Update clinical trials
for query in ["creatine", "muscle", "performance"]:
    await integrator.import_clinical_trials(query, max_results=50)
```

### Monthly Updates (Full)
```python
# Re-run full import with latest data
await integrator.run_full_import()
```

## ⚙️ Configuration

### Enable/Disable APIs

```python
# In src/api_enhanced_rag.py
rag = APIEnhancedRAG(
    use_apis=True,      # Set False to disable
    cache_hours=24      # Cache duration
)
```

### Adjust Rate Limits

```python
# In src/api_integration_config.py
API_CONFIGS["pubmed"].rate_limit = 10.0  # With API key
```

### Custom Queries

```python
# Add domain-specific queries
CREATINE_QUERIES["pubmed"].append("creatine AND mitochondria")
```

## 🐛 Troubleshooting

### "Connection refused" to APIs
- Check internet connection
- Some institutions block certain APIs
- Try using a VPN if needed

### Slow API responses
- Normal for large queries
- Reduce max_results parameter
- Enable caching (default: on)

### Neo4j import errors
- Ensure Neo4j is running: `docker ps`
- Check memory: Neo4j needs ~2GB
- Clear and restart: `docker-compose down && docker-compose up`

## 🚀 Advanced Features

### 1. Combine with Research Mode

When research mode triggers, it can now:
- Validate hypotheses against latest trials
- Find supporting/contradicting evidence
- Generate innovations based on recent findings

### 2. Knowledge Gap Detection

The system identifies when:
- Local knowledge is outdated
- New trials contradict old data
- Emerging research areas appear

### 3. Evidence Scoring

Results are ranked by:
- Recency (newer = higher score)
- Trial phase (Phase 3 > Phase 2 > Phase 1)
- Publication impact (citation count)
- Source reliability

## 📈 Expected Benefits

1. **Accuracy**: Claims backed by specific trials/papers
2. **Currency**: Access to research from last few days
3. **Completeness**: Multiple sources validate findings
4. **Traceability**: Every claim linked to evidence

## 🔮 Future Enhancements

1. **More APIs**:
   - UMLS (requires registration)
   - IntAct (protein interactions)
   - OpenAlex (citation networks)

2. **Smart Caching**:
   - Predictive pre-fetching
   - Differential updates
   - Compressed storage

3. **API Orchestration**:
   - Query planning optimization
   - Cost-based API selection
   - Fallback strategies

## 💡 Best Practices

1. **Start Small**: Import phase 1 first, test, then continue
2. **Monitor Usage**: Stay within rate limits
3. **Cache Wisely**: 24-hour cache balances freshness/performance
4. **Validate Data**: Cross-reference multiple sources
5. **Document Sources**: Always include NCT IDs, PMIDs in responses

This integration transforms your local AI into a connected research system with access to the world's biomedical knowledge while maintaining privacy and local-first principles! 