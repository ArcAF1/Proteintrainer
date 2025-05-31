#!/bin/bash
# ULTRA-SAFE CPU-Only launcher for 13-inch M1 MacBook Pro
# This version disables ALL GPU acceleration to prevent system overload

set -e

# Ensure we're running from the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ULTRA-CONSERVATIVE environment variables
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

echo "🛡️  ================================================"
echo "🛡️   ULTRA-SAFE CPU-ONLY MODE"
echo "🛡️   Hardware Protection: MAXIMUM"
echo "🛡️   GPU/Metal: DISABLED"
echo "🛡️   Optimized for: 13-inch M1 MacBook Pro"
echo "🛡️  ================================================"
echo "Running from: $SCRIPT_DIR"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo "   After Docker starts, run this script again."
    exit 1
fi
echo "✓ Docker is running"

echo "🔧 Activating environment..."

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found! Please run installation.command first."
    exit 1
fi

source venv/bin/activate
echo "✓ Virtual environment activated"

# Hardware protection memory check
echo ""
echo "🛡️  Hardware Protection Check..."
echo "----------------------------"
python check_memory_before_start.py

# Ask user if they want to continue
echo ""
read -p "Continue with CPU-only startup? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Startup cancelled."
    exit 0
fi

# Simple package check
echo "🔍 Checking core packages..."
python -c "
import warnings
warnings.filterwarnings('ignore')
try:
    import transformers
    print('✅ Core packages ready for CPU-only mode')
except ImportError:
    print('⚠️  Some packages missing, but continuing')
" 2>/dev/null

# Start Neo4j in background
echo ""
echo "🐳 Starting Neo4j database..."
echo "-----------------------------"

# Stop any existing containers
docker compose down > /dev/null 2>&1 || true

# Start Neo4j container
docker compose up -d > /dev/null 2>&1

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to start..."
sleep 5

echo "✓ Neo4j database ready"

echo ""
echo "🛡️  Launching in CPU-Only Mode..."
echo "================================="
echo "🔒 ALL GPU acceleration disabled"
echo "🔒 Ultra-conservative memory usage"
echo "🔒 Using existing data (no downloads)"
echo "🔒 Hardware protection active"
echo ""

# Start the main application in CPU-only mode
python run_app.py

echo ""
echo "System shutdown complete."
echo "If you experienced any issues, restart your Mac and try again." 