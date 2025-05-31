# 🧪 Experimental Research Engine Guide

## Overview

This is a **minimal overhead, maximum output** research system designed for rapid experimentation. No red tape, no clinical trial protocols - just fast hypothesis generation and testing.

## Key Features

### ✅ What It Does
- **4-Phase Loop**: Question → Hypothesis → Test → Innovation
- **Uses Your 11GB Data**: Mines your existing PubMed, clinical trials, and pharma data
- **Simple Simulations**: Training response, supplement effects, recovery dynamics
- **Rapid Innovation**: Generates 10-20 ideas per research session
- **Minimal Logging**: Just enough to track progress, not slow you down

### ❌ What It Doesn't Do
- No complex clinical trial design
- No IRB protocols
- No external API calls (uses local data only)
- No complex quality checks
- No formal peer review simulation

## Usage

### Via GUI
1. Go to the **"🧪 Experimental Research"** tab
2. Enter your research question
3. Set iterations (5-10 is good for quick results)
4. Click "Start Experimental Research"
5. Watch innovations appear in real-time

### Via Code
```python
from src.experimental_research_engine import ExperimentalResearchEngine

engine = ExperimentalResearchEngine()
findings = await engine.research_loop(
    "How to maximize muscle protein synthesis?", 
    max_iterations=5
)
```

## Example Output

**Question**: "How to maximize muscle protein synthesis?"

**Iteration 1**:
- Hypothesis: Leucine threshold of 3g triggers maximal mTOR activation
- Evidence: 10 sources from local database
- Simulation: Optimal dose = 3.2g
- Innovations: 
  1. Pre-workout leucine pulse protocol
  2. Combined with resistance training timing
  3. Enhanced absorption with black pepper extract

**Iteration 2**:
- Hypothesis: Training volume correlates with protein synthesis duration
- Evidence: 8 sources found
- Simulation: Peak at 12-16 sets per muscle
- Innovations:
  1. Volume cycling for continuous adaptation
  2. Micro-dosing protein every 2 hours
  3. Sleep optimization for overnight synthesis

## Simulations Available

### 1. Training Response (Banister Model)
- Predicts fitness/fatigue over 8 weeks
- Identifies optimal training load
- Suggests deload timing

### 2. Supplement Dose-Response
- Finds optimal dosing
- Balances benefits vs side effects
- Sigmoid response curves

### 3. Recovery Dynamics
- Models fatigue decay
- Calculates time to full recovery
- Recommends training frequency

## Tips for Maximum Output

1. **Ask Specific Questions**
   - Good: "How to enhance creatine absorption?"
   - Bad: "How to get stronger?"

2. **Let It Run**
   - 5-10 iterations = good balance
   - More iterations = more innovative ideas

3. **Mine the Innovations**
   - Each run generates 10-20 ideas
   - Pick the best 2-3 to investigate deeper

4. **Stack Questions**
   - Run multiple related questions
   - Cross-pollinate ideas between runs

## Integration with Existing Systems

- **Uses Your RAG**: Searches your indexed knowledge
- **GraphRAG Compatible**: Explores entity relationships if available
- **Neo4j Aware**: Can traverse your knowledge graph
- **Works Alongside**: Other research modules without conflict

## Performance

- **Speed**: 1-2 minutes per iteration
- **Memory**: ~500MB additional
- **Output**: 50-100 innovations per hour
- **No External Calls**: Everything runs locally

## Example Research Session

```
1. Start with broad question
2. Get 5-10 hypotheses + innovations
3. Pick most interesting innovation
4. Use it as next question
5. Repeat for deeper insights
```

This creates a **research spiral** that goes deeper with each cycle while maintaining breadth through innovation generation.

## The Philosophy

Traditional research systems focus on rigor and validation. This system focuses on **ideation and discovery**. It's designed to generate hundreds of testable ideas quickly, letting you pick the gems for deeper investigation.

Think of it as your **research idea generator** - feeding your creativity rather than constraining it with protocols. 