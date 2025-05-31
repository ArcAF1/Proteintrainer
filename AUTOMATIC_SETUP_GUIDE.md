# 🚀 Automatic Setup Guide

## What Happens Automatically

When you run `./start_optimized.command`, the system automatically:

### ✅ First Run Setup (Automatic)
- **Detects first run** and applies M1 optimized configuration
- **Enables Metal GPU** acceleration (5-10x faster)
- **Allocates optimal memory** (10GB for best performance)
- **Configures 24 GPU layers** for maximum speed
- **No manual commands needed!**

### ✅ Every Run (Automatic)
- Checks your performance mode
- Sets appropriate environment variables
- Starts Neo4j database
- Selects best model for available memory
- Activates virtual environment

## What You Need to Do

### 🌐 Enable API Enhancement (1 Click)
After the GUI opens:
1. Click **Advanced Tools** (bottom of screen)
2. Toggle **🌐 Enable Live API Data**
3. That's it! Now you get latest research citations

### 🎯 Start Using It!
Try these:
- "What are the latest clinical trials on creatine?"
- "Research muscle recovery optimization"
- "Show me recent studies on beta-alanine"

## That's It! 

Everything else is automatic. No need to:
- ❌ Run `python switch_performance_mode.py` (automatic on first run)
- ❌ Configure GPU settings (automatic)
- ❌ Import ontologies (not needed)
- ❌ Set up complex graphs (your 11GB data is enough)

## If You Want to Change Modes Later

Only if needed:
```bash
# Switch to ultra-fast mode
python switch_performance_mode.py ultra

# Switch to conservative mode (less memory)
python switch_performance_mode.py conservative

# Back to optimized (default)
python switch_performance_mode.py optimized
```

Then restart with `./start_optimized.command`

## Summary

1. **Run:** `./start_optimized.command`
2. **Click:** Enable API in GUI
3. **Use:** Ask your questions!

That's all! 🎉 