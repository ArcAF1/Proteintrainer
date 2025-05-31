# MacBook Pro 13-inch M1 Optimization Guide

## 🍎 Your Setup
- **Hardware**: MacBook Pro 13-inch (2020)
- **Processor**: Apple M1 
- **Memory**: 16GB Unified Memory
- **Usage**: Running other applications alongside AI system

## 🎯 Optimizations Applied

### Memory Management
- **AI System Limit**: ≤ 3GB (leaves 13GB for other apps)
- **Neo4j Database**: Limited to 1GB memory
- **Reserve**: 2GB always kept free for system stability
- **Smart Detection**: Automatically detects other running applications

### CPU Optimization
- **Threads**: Single-threaded operation to avoid conflicts
- **GPU**: Completely disabled to prevent screen flickering
- **Metal**: All Metal/MPS acceleration disabled
- **Priority**: Background processing to not interfere with other apps

### Configuration Files
- `macbook_config.json`: Hardware-specific settings
- `docker-compose.yml`: Neo4j memory limits optimized for MacBook
- `start_macbook_optimized.command`: MacBook-specific launcher

## 🚀 How to Start

### Recommended: MacBook-Optimized Mode
```bash
./start_macbook_optimized.command
```

**Features:**
- ✅ Detects your MacBook automatically
- ✅ Shows memory usage of other apps
- ✅ Uses only 3GB max memory
- ✅ CPU-only mode (no GPU conflicts)
- ✅ Works alongside other applications

### Ultra-Safe Mode (if still having issues)
```bash
./start_cpu_only.command
```

**Features:**
- ✅ Even more conservative settings
- ✅ Uses only 2GB max memory
- ✅ Minimal system impact

## 📊 Performance Expectations

### Response Times
- **Simple queries**: 5-15 seconds
- **Complex analysis**: 15-45 seconds
- **Data processing**: 1-5 minutes

### Memory Usage During Operation
- **AI Models**: 1-2GB
- **Neo4j Database**: 0.5-1GB  
- **Data Processing**: 0.5GB
- **System Overhead**: 0.5GB
- **Total**: ~3GB maximum

### What You Can Run Simultaneously
✅ **Safe to run with AI system:**
- Web browsers (with reasonable tab count)
- Code editors (VS Code, Xcode)
- Terminal applications
- Music/video streaming
- Note-taking apps
- Email clients

⚠️ **May cause resource conflicts:**
- Video editing software
- Heavy IDEs with large projects
- Multiple browsers with 50+ tabs
- Virtual machines
- Games
- Other AI/ML applications

## 🛡️ Hardware Protection Features

### Automatic Detection
- Detects MacBook Pro 13-inch M1 automatically
- Shows other running applications
- Calculates available memory dynamically
- Warns if memory usage is too high

### Safety Mechanisms
- **Memory Threshold**: Stops if >85% memory used
- **GPU Disabled**: Prevents graphics conflicts
- **Resource Limits**: Docker containers have hard limits
- **Graceful Degradation**: Reduces quality instead of crashing

### Monitoring
- Pre-startup memory check
- Real-time application detection
- Post-shutdown memory report

## 🔧 Troubleshooting

### Screen Flickering/Slowdown (FIXED)
This was caused by aggressive GPU usage. Now completely resolved with:
- All GPU/Metal acceleration disabled
- Ultra-conservative memory settings
- Single-threaded operation

### High Memory Usage
```bash
# Check what's using memory
python check_memory_before_start.py

# Recommendations will be shown automatically
```

### Performance Too Slow
1. **Close unnecessary applications**
2. **Restart your MacBook** (clears memory caches)
3. **Use smaller model** (Mistral-7B instead of Medicine-13B)
4. **Increase memory limit** in `macbook_config.json` if you have free RAM

### Docker Issues
```bash
# Restart Docker Desktop
# From Applications folder or menu bar

# Verify Docker is running
docker info
```

## ⚙️ Customization

### Increase Memory Limit (if you have free RAM)
Edit `macbook_config.json`:
```json
{
  "memory_limits": {
    "max_app_memory_gb": 6,        // Increase from 4 to 6
    "reserve_for_other_apps_gb": 2  // Decrease if needed
  }
}
```

### Use Smaller Model for Speed
```bash
# Edit src/llm.py to prefer Mistral over Medicine LLM
# Or delete the larger model file to force smaller one
```

## 📈 Performance Comparison

| Mode | Memory Usage | Speed | Stability | Other Apps |
|------|-------------|-------|-----------|------------|
| **MacBook Optimized** | ~3GB | Medium | Excellent | ✅ Full support |
| **CPU Only** | ~2GB | Slower | Maximum | ✅ Full support |
| **Standard Mode** | ~8GB | Fast | Poor on M1 | ❌ Conflicts |

## 🔍 System Requirements Met

✅ **CPU**: M1 chip with 8 cores  
✅ **Memory**: 16GB (using only 3GB)  
✅ **Storage**: 11GB+ biomedical data already downloaded  
✅ **Docker**: Optimized Neo4j configuration  
✅ **Python**: Virtual environment with all packages  
✅ **Models**: Medicine LLM + Mistral models available  

## 🆘 Support

If you still experience issues:

1. **Try the ultra-safe mode**: `./start_cpu_only.command`
2. **Restart your MacBook** to clear memory
3. **Check Activity Monitor** for memory-hungry apps
4. **Close browser tabs** and unnecessary applications

The system is now specifically optimized for your hardware and usage pattern! 