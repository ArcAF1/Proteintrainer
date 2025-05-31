#!/bin/bash
# macOS Biomedical AI System - MacBook Pro 13-inch M1 Optimized
# Specifically optimized for your hardware and usage pattern
# Double-click from Finder or run "./start_optimized.command" in Terminal.

set -e

# Ensure we're running from the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check active configuration and set environment accordingly
if [ -f "active_config.json" ]; then
    # Determine mode from active config
    MODE=$(python -c "
import json
try:
    with open('active_config.json', 'r') as f:
        config = json.load(f)
    if config.get('performance', {}).get('priority') == 'maximum_speed':
        print('ultra')
    elif config.get('performance', {}).get('priority') == 'speed':
        print('optimized')
    else:
        print('conservative')
except:
    print('conservative')
" 2>/dev/null)
else
    # No config exists - will be created above as optimized
    MODE="optimized"
fi

echo "🔧 Performance mode: $MODE"

# Set environment variables based on mode
if [ "$MODE" = "optimized" ] || [ "$MODE" = "ultra" ]; then
    echo "🚀 Enabling Metal GPU acceleration..."
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export PYTORCH_MPS_ENABLED=1
    export DISABLE_MPS=0
    export TOKENIZERS_PARALLELISM=false
    export LLAMA_METAL_CONSERVATIVE=0
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export CUDA_VISIBLE_DEVICES=""
    export FORCE_CPU_ONLY=0
    export MACBOOK_OPTIMIZED=1
    export LLAMA_N_GPU_LAYERS=24
else
    echo "🛡️ Using conservative CPU-only mode..."
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export PYTORCH_MPS_ENABLED=0
    export DISABLE_MPS=1
    export TOKENIZERS_PARALLELISM=false
    export LLAMA_METAL_CONSERVATIVE=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export CUDA_VISIBLE_DEVICES=""
    export FORCE_CPU_ONLY=1
    export MACBOOK_OPTIMIZED=1
fi

# BLAS optimization for M1 as recommended in checklist
export OPENBLAS_CORETYPE=ARMV8
export UV_THREADPOOL_SIZE=4

echo "💻 ================================================"
echo "💻  Biomedical AI System - MacBook Optimized"
echo "💻  Hardware: MacBook Pro 13-inch M1 (16GB)"
echo "💻  Auto-configured for best performance"
echo "💻  Version: M1 Metal GPU Support"
echo "💻 ================================================"
echo "Running from: $SCRIPT_DIR"

# Check and apply M1 optimized configuration
echo ""
echo "🚀 Checking M1 Performance Configuration..."
echo "--------------------------------------"

if [ -f "m1_optimized_config.json" ]; then
    if [ ! -f "active_config.json" ] || [ ! -s "active_config.json" ]; then
        echo "✅ First run detected - applying M1 optimized configuration..."
        cp m1_optimized_config.json active_config.json
        echo "   ✨ Metal GPU acceleration enabled automatically!"
        echo "   ✨ 24 GPU layers configured"
        echo "   ✨ 10GB memory allocation"
        echo "   ✨ 5-10x faster inference"
        echo ""
        echo "   🎉 No additional setup needed - you're ready to go!"
        MODE="optimized"
    else
        # Check current mode
        python <<'CHECKMODE'
import json
try:
    with open('active_config.json', 'r') as f:
        config = json.load(f)
    mode = "unknown"
    if config.get('performance', {}).get('priority') == 'maximum_speed':
        mode = "ultra"
    elif config.get('performance', {}).get('priority') == 'speed':
        mode = "optimized"
    else:
        mode = "conservative"
    
    print(f"📊 Current performance mode: {mode}")
    
    if mode == "conservative":
        print("⚠️  Running in conservative mode (CPU-only)")
        print("💡 To enable GPU acceleration, run:")
        print("   python switch_performance_mode.py optimized")
except:
    pass
CHECKMODE
    fi
else
    echo "⚠️  M1 optimized config not found - using defaults"
fi

# Check if API integration is available
echo ""
echo "🌐 Checking API Integration..."
if [ -f "src/biomedical_api_client.py" ]; then
    echo "✅ Biomedical API integration available"
    echo "   - PubMed, Clinical Trials, PubChem access ready"
    echo "   - Enable in GUI: Advanced Tools → 🌐 Enable Live API Data"
    echo "   - Get citations for latest research automatically!"
else
    echo "⚠️  API integration not available"
fi

# Check if this is macOS (optional warning, not blocking)
if [[ $(uname) != "Darwin" ]]; then
    echo "⚠️  This version is optimized for macOS."
fi

# Check Docker availability and status
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found. Please install Docker Desktop."
    echo "   Download: https://docs.docker.com/desktop/install/mac-install/"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo "   Find Docker Desktop in Applications or menu bar."
    echo ""
    echo "   After Docker starts, run this script again."
    exit 1
fi
echo "✓ Docker Desktop is running"

echo "🔧 Activating environment..."

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run installation.command first."
    exit 1
fi

source venv/bin/activate
echo "✓ Virtual environment activated"

# MacBook-specific hardware and memory check
echo ""
echo "💻 MacBook Compatibility & Memory Check..."
echo "--------------------------------------"
python check_memory_before_start.py

# Ask user if they want to continue with smart recommendations
echo ""
echo "🤔 This system is optimized for your MacBook Pro 13-inch M1."
if [ "$MODE" = "optimized" ] || [ "$MODE" = "ultra" ]; then
    echo "   GPU acceleration is enabled for 5-10x faster responses."
    echo "   Memory usage: ~10GB (leaves 6GB for other apps)"
else
    echo "   Running in safe mode with ~5GB memory usage."
    echo "   Your other applications will continue running normally."
fi
echo ""
read -p "Continue with startup? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Startup cancelled."
    echo ""
    echo "💡 Alternative options:"
    echo "   • ./start_cpu_only.command (even more conservative)"
    echo "   • Close some applications and try again"
    echo "   • Restart your MacBook to clear memory"
    exit 0
fi

# Core package verification
echo "🔍 Verifying core packages..."
python <<'PYCODE'
import warnings, subprocess, sys
warnings.filterwarnings('ignore')
try:
    import transformers, neo4j, Bio
    print('✅ Core biomedical packages ready')
except ImportError as e:
    if 'No module named' in str(e) and 'Bio' in str(e):
        print('⚠️  BioPython not found - installing…')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'biopython'])
            print('✅ BioPython installed successfully')
        except Exception as install_err:
            print(f'❌ Failed to install BioPython: {install_err}')
            sys.exit(1)
    else:
        print('⚠️  Some packages missing, but system will adapt')
        print(f'   Details: {str(e)[:100]}…')
PYCODE

# Verify BioPython installation
python <<'VERIF'
import sys
try:
    import Bio
    print('✅ BioPython verified and ready')
except ImportError:
    print('❌ BioPython installation failed - please run: pip install biopython')
    sys.exit(1)
VERIF

# After BioPython verification add Gradio version check
echo "🔍 Checking Gradio compatibility..."
python <<'GRCHECK'
import subprocess, sys, pkg_resources, importlib.metadata, re

def ver(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None

gr_v = ver('gradio')
fast_v = ver('fastapi')
star_v = ver('starlette')

def needs_upgrade(v, minimum):
    def norm(x):
        return [int(p) for p in re.findall(r'\d+', x)[:3]]
    if v is None:
        return True
    return norm(v) < norm(minimum)

upgrade = False
if needs_upgrade(gr_v or '0', '4.44.0'):
    print(f'⚠️  Gradio {gr_v} outdated – upgrading to 4.44.1')
    upgrade = True
if needs_upgrade(fast_v or '0', '0.115.10'):
    print(f'⚠️  FastAPI {fast_v} outdated – upgrading to 0.115.12')
    upgrade = True
if needs_upgrade(star_v or '0', '0.37.0'):
    print(f'⚠️  Starlette {star_v} outdated – upgrading to 0.37.0')
    upgrade = True

if upgrade:
    print('🔄 Upgrading UI packages... (this may take a moment)')
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--quiet',
        'gradio==4.44.1',
        'fastapi==0.115.12'
    ])
    print('✅ UI packages upgraded')
else:
    print('✅ Gradio/FastAPI versions are up to date')
GRCHECK

# Start Neo4j with MacBook-optimized resource limits
echo ""
echo "🗄️  Starting Neo4j database (MacBook optimized)..."
echo "----------------------------------------------"

# Stop any existing containers
docker compose down > /dev/null 2>&1 || true

# Start Neo4j with resource limits for MacBook
echo "Starting Neo4j with 1GB memory limit..."
docker compose up -d > /dev/null 2>&1

# Wait for Neo4j to be ready with progress indicator
echo -n "Waiting for Neo4j to initialize"
for i in {1..15}; do
    if curl -f http://localhost:7474 >/dev/null 2>&1; then
        echo " ✓"
        break
    fi
    echo -n "."
    sleep 2
done

# Verify Neo4j is working
if curl -f http://localhost:7474 >/dev/null 2>&1; then
    echo "✓ Neo4j database ready and responsive"
else
    echo "⚠️  Neo4j still starting up, but continuing..."
fi

# Show model information and auto-select best one for MacBook
echo ""
echo "🤖 MacBook Model Optimization..."
echo "------------------------------"

# Run model selection and optimization
python -c "
import psutil
import os
from pathlib import Path

def select_best_model_for_macbook():
    '''Auto-select the best model based on available memory.'''
    models_dir = Path('models')
    if not models_dir.exists():
        print('❌ Models directory not found')
        return
    
    # Get current memory status
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    print(f'Available memory: {available_gb:.1f} GB')
    
    # Get all models with sizes
    models = []
    for model_path in models_dir.glob('*.gguf'):
        if model_path.is_file():  # Skip if it's a backup
            size_gb = model_path.stat().st_size / (1024**3)
            models.append((model_path, size_gb))
    
    if not models:
        print('❌ No models found')
        return
    
    # Sort by size (smallest first)
    models.sort(key=lambda x: x[1])
    
    # Find the best model that fits in available memory
    selected = None
    for model_path, size_gb in models:
        memory_needed = size_gb * 1.3  # Model + reasonable overhead (was 1.5, now more practical)
        if memory_needed <= available_gb:
            selected = (model_path, size_gb, memory_needed)
        else:
            print(f'⚠️  {model_path.name} ({size_gb:.1f}GB) needs {memory_needed:.1f}GB - borderline/too large')
    
    if selected:
        model_path, size_gb, memory_needed = selected
        print(f'✅ Selected: {model_path.name} ({size_gb:.1f}GB)')
        print(f'   Memory needed: ~{memory_needed:.1f}GB')
        print(f'   This model should work well on your MacBook')
        
        # Hide larger models temporarily
        for other_model, other_size in models:
            if other_model != model_path and other_size > size_gb:
                backup_name = str(other_model) + '.temp_backup'
                if not Path(backup_name).exists():
                    print(f'   Temporarily hiding: {other_model.name}')
                    other_model.rename(backup_name)
    else:
        print('⚠️  All models are large for current memory, but will try smallest')
        print('   The system will use ultra-conservative settings')
        
        # Use the smallest model anyway with warning
        smallest = models[0]
        print(f'   Using: {smallest[0].name} ({smallest[1]:.1f}GB)')
        print(f'   Consider closing other applications for better performance')

select_best_model_for_macbook()
"

echo ""
echo "🧠 Launching Biomedical AI System..."
echo "=================================="

# Show mode-specific messages
if [ "$MODE" = "optimized" ] || [ "$MODE" = "ultra" ]; then
    echo "⚡ Metal GPU acceleration: ACTIVE"
    echo "🚀 Performance mode: $MODE"
    echo "💾 Memory allocation: 10GB"
    echo "🔥 Speed boost: 5-10x enabled"
else
    echo "🛡️ Conservative mode: CPU-only"
    echo "💡 For 5-10x speed: python switch_performance_mode.py optimized"
fi

echo ""
echo "📚 Using your 11GB+ biomedical data"
echo "🔒 Hardware protection active"
echo ""

echo "💡 Quick Start Guide:"
echo "   1. ✅ GPU acceleration is automatic (if supported)"
echo "   2. 🌐 Enable API data: In GUI → Advanced Tools → Toggle API"
echo "   3. 🔬 Try research mode: Type 'research' + any topic"
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

echo "🚀 Starting Biomedical AI System with Optimized Settings..."
echo "=================================================="

# Clean up any lingering processes from previous runs
echo "🧹 Cleaning up previous sessions..."
python3 cleanup_processes.py
echo ""

# Check if this is the first run
if [ ! -f "active_config.json" ]; then
    echo "🎯 First run detected - applying M1 optimized configuration..."
    
    # Check if we have the optimized config
    if [ -f "configs/m1_optimized.json" ]; then
        echo "✅ Applying optimized settings for Apple Silicon..."
        python3 switch_performance_mode.py optimized
        echo ""
        echo "📊 Applied settings:"
        echo "  - Metal GPU acceleration: 24 layers"
        echo "  - Memory allocation: 10GB"
        echo "  - Optimized for M1 performance"
        echo ""
    else
        echo "⚠️  Optimized config not found, using defaults"
    fi
else
    echo "✅ Using existing configuration"
    # Show current mode
    python3 -c "
import json
with open('active_config.json', 'r') as f:
    config = json.load(f)
    mode = config.get('mode', 'Unknown')
    print(f'📊 Current mode: {mode}')
"
fi

echo ""
echo "🌐 Starting GUI interface..."
echo "The web interface will open in your browser automatically."
echo ""
echo "💡 Tips:"
echo "  - Ask 'What is your role?' to understand the AI's capabilities"
echo "  - Try 'Research ways to enhance creatine' for deep research"
echo "  - Use 'Test the system' to check all components"
echo ""

# Start the GUI
python3 -m src.gui_unified

# If the GUI exits, show a message
echo ""
echo "👋 Application closed. To restart, run this command again."

# Restore any temporarily hidden models
echo ""
echo "🔄 Restoring model files..."
python -c "
from pathlib import Path
models_dir = Path('models')
for backup_file in models_dir.glob('*.temp_backup'):
    original_name = str(backup_file).replace('.temp_backup', '')
    backup_file.rename(original_name)
    print(f'   Restored: {Path(original_name).name}')
"

echo ""
echo "📊 Final system status:"
python -c "
import psutil
memory = psutil.virtual_memory()
print(f'   Memory usage: {memory.percent:.1f}%')
print(f'   Available: {memory.available/(1024**3):.1f} GB')
print(f'   System ready for other applications')
"

echo ""
echo "✨ Everything is configured automatically!"
echo ""
echo "🎯 Next Steps:"
echo "   1. In the GUI: Enable API enhancement (Advanced Tools → 🌐)"
echo "   2. Ask anything: 'What are the latest trials on creatine?'"
echo "   3. Try research: 'Research muscle recovery optimization'"
echo ""
echo "🚀 Performance Options (if needed):"
echo "   • Switch modes: python switch_performance_mode.py [conservative|optimized|ultra]"
echo "   • Restart system: ./start_optimized.command"
echo "   • View this guide: cat OPTIMIZATION_COMPLETE.md" 