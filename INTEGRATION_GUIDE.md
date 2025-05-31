# 🔧 Integration Guide - Medical Research System

## Quick Integration Steps

### 1. Install New Dependencies
```bash
# Install medical research requirements
pip install chromadb arxiv requests

# Or use the full requirements file
pip install -r requirements_medical.txt
```

### 2. Run Setup
```bash
python setup_medical_research.py
```

### 3. Update Your Code

If you're using the system programmatically, update your imports:

```python
# Old way (outputs sources first)
from src.rag_chat import answer

# New way (outputs answers first, does research)
from src.enhanced_rag_chat import answer_medical_query

# Use it the same way
response = await answer_medical_query("What is metformin?")
```

### 4. The GUI Already Works!

Your existing GUI (`src.gui_unified.py`) is already updated to use the medical research system. Just start it:

```bash
python -m src.gui_unified
```

## What's Changed

### Before:
- LLM would output: "Sources: [1] Paper A [2] Paper B..."
- No automatic research
- Fixed knowledge base

### After:
- LLM outputs: "Metformin is a medication that works by..."
- Automatic PubMed/ArXiv searches
- Growing knowledge base
- Confidence scores

## Testing the Changes

1. **Test Output Format:**
   ```
   Ask: "What is the mechanism of metformin?"
   Should get: Answer first, then sources
   ```

2. **Test Research:**
   ```
   Ask: "What are the latest 2024 advances in longevity?"
   Should see: "🔬 Research Update: I conducted new research..."
   ```

3. **Test Alternatives:**
   ```
   Ask: "What's better than creatine?"
   Should get: List of alternatives with explanations
   ```

## Configuration Options

Edit `medical_config.json` to customize:

```json
{
  "research_auto_trigger": true,    // Auto-search when gaps detected
  "knowledge_coverage_threshold": 0.7,  // When to trigger research
  "confidence_display": true         // Show confidence scores
}
```

## Minimal Code Example

```python
import asyncio
from src.enhanced_rag_chat import answer_medical_query

async def main():
    # Ask any medical question
    response = await answer_medical_query(
        "What gives the same effect as creatine but is legal and better?"
    )
    print(response)

asyncio.run(main())
```

## That's It! 🎉

Your system now:
- ✅ Answers questions properly (not just lists sources)
- ✅ Conducts research automatically
- ✅ Learns from each query
- ✅ Provides confidence scores

The medical research agent runs seamlessly in the background! 