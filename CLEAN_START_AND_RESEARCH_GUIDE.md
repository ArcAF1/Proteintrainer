# 🚀 Clean Start & Deep Research Guide

## Clean Start Mechanism

Your system now includes automatic cleanup to prevent multiple LLM instances and ensure smooth operation.

### What Gets Cleaned Up

1. **Previous Python Processes**
   - Any lingering GUI instances
   - Stuck LLM processes
   - Orphaned embedder processes
   - High-memory processes (>2GB)

2. **Ports**
   - Frees ports 7860-7865 for Gradio

3. **Lock Files**
   - Removes stale lock files that could cause conflicts

4. **Shared Memory** (macOS)
   - Clears shared memory segments

### How to Use

Just run the start script as normal:
```bash
./start_optimized.command
```

The cleanup happens automatically before starting the GUI.

### Manual Cleanup

If needed, you can run cleanup manually:
```bash
python3 cleanup_processes.py
```

## Enhanced Research System

The research engine now runs for appropriate durations to ensure thorough investigation.

### Research Depths

The system automatically detects research depth from your query:

1. **Quick Research** (5-10 minutes)
   - Triggered by: "quick research on...", "brief summary of..."
   - 5 iterations maximum
   - Good for quick overviews

2. **Standard Research** (10-20 minutes)
   - Default for most research queries
   - 10 iterations
   - Balanced depth and speed

3. **Deep Research** (20-40 minutes)
   - Triggered by: "deep research on...", "comprehensive analysis of...", "thorough investigation of..."
   - 20 iterations minimum
   - Thorough exploration with detailed analysis

4. **Comprehensive Research** (30-60 minutes)
   - For maximum depth
   - 30 iterations
   - Exhaustive investigation

### Research Features

Each iteration now includes:
- **Minimum 1 minute per iteration** to ensure quality
- **Thinking pauses** between phases
- **Progress logging** with timestamps
- **Duration tracking** in final report

### Example Queries

**Quick Research:**
```
"Do quick research on creatine timing"
```

**Standard Research:**
```
"Research ways to enhance creatine effectiveness"
```

**Deep Research:**
```
"Do deep research on creatine and what could enhance it"
"Comprehensive analysis of muscle recovery optimization"
```

### What to Expect

1. **Research Start**
   - Shows estimated duration
   - Indicates research depth
   - Begins with hypothesis generation

2. **During Research**
   - Each iteration takes ~1-3 minutes
   - Progress logged to research file
   - System searches knowledge base, runs simulations, generates innovations

3. **Research Complete**
   - Shows total duration
   - Lists top 10 innovations
   - Provides research quality indicator
   - Full log saved to `experimental_research/research_log.md`

### Research Quality Indicators

- 🏆 **Deep Research**: 20+ minutes of comprehensive investigation
- 📊 **Standard Research**: 10-20 minutes of solid analysis  
- ⚡ **Quick Research**: 5-10 minutes of rapid exploration

## Tips for Best Results

1. **Memory Management**
   - The cleanup ensures only one LLM instance runs
   - Your system can handle 30+ minute research sessions
   - All data (chat memory, research logs) is preserved

2. **Research Queries**
   - Use "deep" or "comprehensive" for thorough analysis
   - Default research takes 10-20 minutes
   - Check the log file for detailed findings

3. **System Health**
   - Run "test the system" to check components
   - Cleanup runs automatically on each start
   - Manual cleanup available if needed

## Troubleshooting

**If the app won't start:**
1. Run `python3 cleanup_processes.py`
2. Check if port 7860 is free: `lsof -i :7860`
3. Restart your terminal and try again

**If research seems too quick:**
- Use "deep research" or "comprehensive research" keywords
- Check the log file - it shows actual duration
- Ensure the system has finished (shows completion message)

**If you get memory errors:**
- The cleanup should prevent this
- Check Activity Monitor for high-memory processes
- Restart your Mac if needed

Your system is now optimized for clean starts and proper deep research! 🎉 