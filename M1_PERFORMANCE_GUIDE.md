# 🚀 M1 MacBook Performance Optimization Guide

## Current Issue
Your system is running in **conservative mode** (CPU-only), which is why you're experiencing 200-second response times. We can speed this up by **5-10x** using your M1's GPU!

## Quick Fix - Switch to Optimized Mode

### Option 1: Command Line (Recommended)
```bash
# Switch to optimized mode (best balance)
python switch_performance_mode.py optimized

# Or for maximum speed (when doing dedicated AI work)
python switch_performance_mode.py ultra
```

### Option 2: Manual Config Update
Replace the contents of `macbook_config.json` with `m1_optimized_config.json`

## Performance Modes

### 🐌 Conservative Mode (Current)
- **Speed**: ~200 seconds per response
- **Memory**: 5GB max
- **GPU**: Disabled
- **Use when**: Running many other apps

### 🚀 Optimized Mode (Recommended)
- **Speed**: ~20-40 seconds per response
- **Memory**: 10GB max
- **GPU**: 24 layers on Metal
- **Use when**: Normal daily use with AI

### ⚡ Ultra Mode
- **Speed**: ~10-20 seconds per response
- **Memory**: 14GB max
- **GPU**: 32 layers on Metal
- **Use when**: Dedicated AI sessions only

## What Changes?

1. **Metal GPU Acceleration**: Uses M1's GPU for 5-10x faster inference
2. **Larger Batch Sizes**: Processes more tokens at once
3. **Better Memory Management**: Uses unified memory efficiently
4. **Optimized Threading**: Uses performance cores properly
5. **KV Cache on GPU**: Faster attention computations

## Monitoring Performance

### Check Current Speed
```bash
python switch_performance_mode.py benchmark
```

### Expected Results
- **Conservative**: 2-5 tokens/second
- **Optimized**: 20-40 tokens/second
- **Ultra**: 40-80 tokens/second

## Tips for Best Performance

1. **Close Heavy Apps**: Chrome, Slack, etc. use lots of memory
2. **Restart After Switching**: Ensures clean memory state
3. **Use Activity Monitor**: Watch GPU usage in Activity Monitor
4. **Temperature**: Your Mac may get warm - this is normal

## Troubleshooting

### If Crashes Occur
```bash
# Switch back to conservative
python switch_performance_mode.py conservative
```

### If Still Slow
1. Run system test: "Test if the system is working properly"
2. Check if indexes are built: Look for `indexes/pmc.faiss`
3. Ensure model is downloaded: Check `models/` folder

### Memory Pressure
- Yellow memory pressure: Normal with optimized mode
- Red memory pressure: Switch to conservative or close apps

## Research Performance

With optimized mode, research will be MUCH faster:
- **Conservative**: 5-10 minutes per research
- **Optimized**: 1-3 minutes per research
- **Ultra**: 30-90 seconds per research

## Next Steps

1. **Switch to optimized mode now**:
   ```bash
   python switch_performance_mode.py optimized
   ```

2. **Test with a simple question**:
   "What do you know about creatine?"
   
3. **Try research mode**:
   "Research ways to enhance creatine effectiveness"

The difference will be dramatic! 🚀 