#!/bin/bash
# MacBook Pro 13-inch M1 Optimized launcher
# Specifically designed for 16GB M1 MacBook Pro running other applications
# Prioritizes stability and resource sharing

set -e

# Ensure we're running from the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# MacBook-optimized environment variables
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

echo "💻 ================================================"
echo "💻  MacBook Pro 13-inch M1 Optimized Mode"
echo "💻  Hardware: Apple M1, 16GB Unified Memory"
echo "💻  Mode: Ultra-Conservative + Other Apps Support"
echo "💻  GPU: Completely Disabled for Stability"
echo "💻 ================================================"
echo "Running from: $SCRIPT_DIR"

# Check if this is actually a MacBook
if [[ $(uname) != "Darwin" ]]; then
    echo "⚠️  This script is optimized for macOS. Use start_cpu_only.command instead."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Check Docker availability
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found. Please install Docker Desktop for Mac."
    echo "   Download from: https://docs.docker.com/desktop/install/mac-install/"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo "   You can find Docker Desktop in Applications or in the menu bar."
    echo ""
    echo "   After Docker starts, run this script again:"
    echo "   ./start_macbook_optimized.command"
    exit 1
fi
echo "✓ Docker Desktop is running"

echo "🔧 Activating Python environment..."

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run installation.command first to set up the environment."
    exit 1
fi

source venv/bin/activate
echo "✓ Python environment activated"

# MacBook-specific memory and compatibility check
echo ""
echo "💻 MacBook Compatibility Check..."
echo "--------------------------------"
python check_memory_before_start.py

# Ask user if they want to continue
echo ""
echo "🤔 The AI system will use minimal resources to coexist with your other apps."
echo "   It will be slower but won't interfere with your workflow."
echo ""
read -p "Continue with MacBook-optimized startup? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Startup cancelled. You can also try:"
    echo "• Close some applications to free memory"
    echo "• Restart your MacBook to clear memory"
    echo "• Use ./start_cpu_only.command for even more conservative settings"
    exit 0
fi

# Simple package verification
echo "🔍 Verifying core packages..."
python -c "
import warnings
warnings.filterwarnings('ignore')
try:
    import transformers, neo4j
    print('✅ Core AI packages ready')
except ImportError as e:
    print('⚠️  Some packages missing, but system will adapt')
    print(f'   Details: {e}')
" 2>/dev/null

# Configure and start Neo4j with MacBook-optimized settings
echo ""
echo "🗄️  Starting Neo4j database (MacBook optimized)..."
echo "----------------------------------------------"

# Stop any existing containers gracefully
docker compose down > /dev/null 2>&1 || true

# Start Neo4j with resource limits
echo "Starting Neo4j with 1GB memory limit..."
docker compose up -d > /dev/null 2>&1

# Wait for Neo4j with progress
echo -n "Waiting for Neo4j to initialize"
for i in {1..10}; do
    if docker exec proteintrainer-neo4j cypher-shell -u neo4j -p "BioMed@2024!Research" "RETURN 1" >/dev/null 2>&1; then
        echo " ✓"
        break
    fi
    echo -n "."
    sleep 2
done

# Verify Neo4j is responsive
if docker exec proteintrainer-neo4j cypher-shell -u neo4j -p "BioMed@2024!Research" "RETURN 1" >/dev/null 2>&1; then
    echo "✓ Neo4j database ready and responsive"
else
    echo "⚠️  Neo4j may still be starting up, but continuing..."
fi

echo ""
echo "🧠 Launching Biomedical AI (MacBook Mode)..."
echo "==========================================="
echo "🔒 CPU-only processing (no GPU conflicts)"
echo "🔒 Memory usage: ≤ 3GB (leaves 13GB for other apps)"
echo "🔒 Single-threaded operation"
echo "🔒 Using your existing 11GB+ biomedical data"
echo "🔒 Neo4j memory limited to 1GB"
echo ""
echo "💡 Performance expectations:"
echo "   • Slower than GPU mode, but stable"
echo "   • Won't interfere with other applications"
echo "   • Responses in 10-30 seconds"
echo "   • Perfect for research and learning"
echo ""

# Start the main application with MacBook configuration
python run_app.py

echo ""
echo "💻 MacBook-optimized AI system shutdown complete."
echo ""
echo "📊 System status:"
# Show memory usage after shutdown
python -c "
import psutil
memory = psutil.virtual_memory()
print(f'   Memory usage: {memory.percent:.1f}%')
print(f'   Available: {memory.available/(1024**3):.1f} GB')
"

echo ""
echo "🔄 To restart: ./start_macbook_optimized.command"
echo "💡 For even lighter usage: ./start_cpu_only.command" 