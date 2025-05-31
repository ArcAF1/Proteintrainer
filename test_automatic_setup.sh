#!/bin/bash
# Test that automatic setup works correctly

echo "🧪 Testing Automatic Setup"
echo "========================="

# 1. Check if active_config.json gets created automatically
if [ -f "active_config.json" ]; then
    echo "✅ Active config exists"
    
    # Check if it's optimized mode
    MODE=$(python -c "
import json
with open('active_config.json', 'r') as f:
    config = json.load(f)
if config.get('performance', {}).get('priority') == 'speed':
    print('optimized')
else:
    print('other')
" 2>/dev/null)
    
    if [ "$MODE" = "optimized" ]; then
        echo "✅ Optimized mode is active"
    else
        echo "❌ Not in optimized mode"
    fi
    
    # Check GPU layers
    GPU_LAYERS=$(python -c "
import json
with open('active_config.json', 'r') as f:
    config = json.load(f)
print(config.get('llm', {}).get('n_gpu_layers', 0))
" 2>/dev/null)
    
    if [ "$GPU_LAYERS" = "24" ]; then
        echo "✅ GPU layers set to 24"
    else
        echo "❌ GPU layers not set correctly: $GPU_LAYERS"
    fi
else
    echo "❌ No active config - first run will create it"
    echo "   This is expected if you haven't run start_optimized.command yet"
fi

# 2. Check if m1_optimized_config.json exists
if [ -f "m1_optimized_config.json" ]; then
    echo "✅ M1 optimized config template exists"
else
    echo "❌ M1 optimized config missing"
fi

# 3. Check API files
if [ -f "src/biomedical_api_client.py" ]; then
    echo "✅ API integration files present"
else
    echo "❌ API integration files missing"
fi

echo ""
echo "Summary:"
echo "--------"
echo "• Run ./start_optimized.command"
echo "• Everything else happens automatically!"
echo "• Just enable API in the GUI when it opens" 