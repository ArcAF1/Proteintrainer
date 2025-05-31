# 🏥 Autonomous Medical Research System

## Overview

This is a complete autonomous medical research system for your local Mistral 7B LLM that:
1. **Fixes the output format issue** - Forces the LLM to answer questions first, then list sources
2. **Conducts autonomous research** - Automatically searches PubMed, ArXiv, and ClinicalTrials.gov
3. **Maintains a knowledge base** - Uses ChromaDB for vector storage and continuous learning
4. **Validates knowledge** - Provides confidence scores and validates claims against sources

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements_medical.txt

# Run setup script
python setup_medical_research.py

# Test the system
python test_medical_system.py
```

### 2. Start the GUI
```bash
python -m src.gui_unified
```

### 3. Ask Medical Questions
The system will automatically:
- Answer your question directly (not just list sources)
- Detect knowledge gaps
- Search medical databases
- Update its knowledge base
- Provide confidence scores

## 🔧 Key Features

### 1. Fixed Output Format
The system uses special prompts to ensure Mistral 7B:
- Always answers the question first
- Provides comprehensive explanations
- Lists sources only at the end
- Never starts with citations

**Example Prompt Structure:**
```
[ANSWER]
Your comprehensive answer here...

[SOURCES]
1. Source 1
2. Source 2
```

### 2. Autonomous Research
When you ask a question:
1. Checks existing knowledge (coverage score)
2. If coverage < 70%, automatically searches:
   - **PubMed** - Medical research papers
   - **ArXiv** - Preprints and latest research
   - **ClinicalTrials.gov** - Clinical trial data
3. Downloads and analyzes papers
4. Updates knowledge base
5. Provides comprehensive answer

### 3. Knowledge Base
- Uses **ChromaDB** for vector storage
- **Sentence Transformers** for embeddings
- Persistent storage in `medical_research/chroma_db/`
- Automatic deduplication
- Relevance scoring

### 4. Confidence Scoring
Every answer includes:
- Confidence percentage (0-95%)
- Evidence quality assessment
- Knowledge gap identification
- Source diversity scoring

## 📚 Usage Examples

### Example 1: Creatine Alternatives
```python
# Ask: "What gives the same effect as creatine but is legal and better?"

# System will:
1. Check knowledge about creatine
2. Identify gaps about alternatives
3. Search PubMed for "creatine alternatives"
4. Search for clinical trials
5. Synthesize findings
6. Return answer with alternatives like:
   - Beta-alanine
   - Citrulline malate
   - HMB (β-Hydroxy β-Methylbutyrate)
```

### Example 2: Latest Research
```python
# Ask: "What are the latest breakthroughs in NAD+ supplementation?"

# System will:
1. Detect this requires recent information
2. Search ArXiv and PubMed with date filters
3. Analyze latest papers
4. Provide up-to-date answer
```

### Example 3: Mechanism Questions
```python
# Ask: "What is the mechanism of action of metformin?"

# System will:
1. Provide direct answer about AMPK activation
2. Explain cellular mechanisms
3. Describe clinical effects
4. List scientific sources at the end
```

## 🏗️ Architecture

### Core Components

1. **MedicalResearchAgent** (`src/medical_research_agent.py`)
   - Inherits from AutonomousResearchAgent
   - Integrates medical databases
   - Manages knowledge base
   - Conducts autonomous research

2. **EnhancedRAGChat** (`src/enhanced_rag_chat.py`)
   - Fixes output formatting
   - Routes medical queries
   - Formats responses
   - Handles confidence scoring

3. **MedicalKnowledgeValidator**
   - Validates claims against sources
   - Calculates confidence scores
   - Identifies contradictions

### Data Flow

```
User Query
    ↓
Enhanced RAG Chat
    ↓
Knowledge Coverage Check
    ↓
[If gaps detected]
    ↓
Autonomous Research
    ├── PubMed API
    ├── ArXiv API
    └── ClinicalTrials API
    ↓
Knowledge Base Update
    ↓
Generate Answer
    ↓
Format Response
    ↓
User Gets Answer + Sources
```

## 🔬 Advanced Features

### Custom Prompts
The system uses specialized prompts for Mistral 7B:

```python
MISTRAL_SYSTEM_PROMPT = """You are a medical AI assistant. IMPORTANT RULES:

1. ALWAYS provide a direct answer FIRST
2. NEVER start with sources or citations
3. List sources ONLY at the end of your response
4. Use clear, structured formatting

When answering:
- Start with a comprehensive answer
- Explain mechanisms and effects
- Provide practical recommendations
- End with sources (if any)

Remember: Users want answers, not just references."""
```

### Knowledge Gap Analysis
The system automatically identifies missing information:
- Concept extraction from queries
- Coverage scoring
- Gap identification
- Targeted research queries

### Continuous Learning
- Every research result is saved
- Knowledge base grows over time
- Deduplication prevents redundancy
- Relevance decay for old information

## 🛠️ Configuration

Edit `medical_config.json`:
```json
{
  "mistral_prompt_format": "answer_first",
  "research_auto_trigger": true,
  "knowledge_coverage_threshold": 0.7,
  "max_research_sources": 10,
  "confidence_display": true,
  "pubmed_retmax": 10,
  "arxiv_max_results": 5,
  "clinical_trials_max": 5
}
```

## 📊 Performance Optimization

### For MacBook (4.1GB Model)
- Efficient vector search with FAISS fallback
- Lightweight ChromaDB configuration
- Async processing for non-blocking UI
- Batch embedding generation
- Smart caching of frequent queries

### Memory Usage
- ~500MB for ChromaDB
- ~200MB for embeddings model
- ~4.1GB for Mistral model
- Total: ~5GB RAM usage

## 🐛 Troubleshooting

### Issue: LLM still outputs sources first
**Solution:** The enhanced prompts should fix this, but if not:
1. Check `MISTRAL_SYSTEM_PROMPT` is being used
2. Ensure you're using EnhancedRAGChat
3. Update to latest Mistral model

### Issue: Research takes too long
**Solution:**
1. Reduce `max_research_sources` in config
2. Use async GUI to avoid blocking
3. Check internet connection

### Issue: ChromaDB errors
**Solution:**
```bash
# Reset database
rm -rf medical_research/chroma_db/
python setup_medical_research.py
```

## 🚦 Next Steps

1. **Add More Databases**
   - Google Scholar API
   - bioRxiv
   - PubMed Central full text

2. **Improve NLP**
   - BioBERT for better medical NER
   - scispaCy for entity linking

3. **Enhanced Validation**
   - Cross-reference checking
   - Contradiction detection
   - Evidence grading (GRADE system)

## 📝 Example Session

```
You: What gives the same effect as creatine but is legal and better?

AI: Based on my research, here are several legal alternatives to creatine that can provide similar or complementary benefits:

**Beta-Alanine** is one of the most researched alternatives. It works by increasing muscle carnosine levels, which helps buffer acid in muscles during high-intensity exercise. While creatine primarily helps with strength and power, beta-alanine excels at reducing fatigue during longer sets (30 seconds to 4 minutes). Studies show 3-5g daily can improve muscular endurance by 2-4%.

**Citrulline Malate** enhances nitric oxide production, improving blood flow and nutrient delivery to muscles. Research indicates 6-8g before training can reduce muscle soreness by up to 40% and increase training volume. Unlike creatine's water retention, citrulline doesn't cause weight gain.

**HMB (β-Hydroxy β-Methylbutyrate)** helps preserve muscle mass, especially during caloric deficits. It's particularly effective for beginners or during intense training phases. The recommended dose is 3g daily, split into 1g servings with meals.

For many athletes, combining these supplements may provide better results than creatine alone, as they target different mechanisms of performance enhancement.

✅ **Confidence:** High - based on multiple quality sources

🔬 **Research Update:** I conducted new research to answer your question, filling knowledge gaps about: alternative_treatments, legal supplements

**Sources:**
1. PubMed: "Beta-alanine supplementation to improve exercise capacity and performance: a systematic review and meta-analysis"
2. ArXiv: "Comparative analysis of ergogenic aids in strength training"
3. ClinicalTrials.gov: NCT03490942 - "Effects of Citrulline Supplementation on Exercise Performance"
```

## 🎯 Conclusion

This system transforms your local Mistral 7B into an autonomous medical research assistant that:
- ✅ Answers questions directly (fixes output format)
- ✅ Conducts research automatically
- ✅ Maintains and updates knowledge
- ✅ Provides confidence scores
- ✅ Works efficiently on your MacBook

The AI now acts like a real medical researcher, finding and synthesizing information to answer your questions comprehensively. 