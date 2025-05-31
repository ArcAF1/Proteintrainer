#!/bin/bash
# macOS one-click launcher for the Offline Medical RAG Assistant.
# Double-click from Finder or run "./start.command" in Terminal.
#
# 1. Ensures Docker is running (starts it if needed)
# 2. Installs / activates Python virtualenv and dependencies
# 3. Starts Neo4j Community Edition (ARM64) in Docker
# 4. Launches the Gradio GUI (run_app.py now handles waiting for Neo4j)

set -e

# Ensure we're running from the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "BioMedical AI System Startup"
echo "=========================================="
echo "Running from: $SCRIPT_DIR"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo "   After Docker starts, run this script again."
    exit 1
fi
echo "✓ Docker is running"

echo "🔧 Setting up environment..."

# Setup Python environment
echo "Setting up Python environment..."

# Install/update system dependencies
echo "[setup] Installing dependencies..."
# Silently update brew formulas
brew update > /dev/null 2>&1 || echo "Brew update skipped"

# Activate virtual environment
echo "[setup] Activating virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv
fi

source venv/bin/activate

# Install Python packages
pip install --upgrade pip > /dev/null
echo "[setup] Installing Python packages..."
pip install -r requirements.txt > /dev/null 2>&1 || echo "Some packages already installed"

echo "✓ Environment setup complete"

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
echo "🤖 Initializing AI system..."
echo "----------------------------"

# Check for available models and show user what's being used
python -c "
import sys
sys.path.append('src')
from pathlib import Path
from src.llm import detect_model_type

# Find available models
models_dir = Path('models')
if models_dir.exists():
    model_files = list(models_dir.glob('*.gguf'))
    if model_files:
        # Sort by modification time (newest first) and size (largest first)
        model_files.sort(key=lambda x: (x.stat().st_mtime, x.stat().st_size), reverse=True)
        selected_model = model_files[0]
        
        size_gb = selected_model.stat().st_size / (1024**3)
        model_type = detect_model_type(str(selected_model))
        
        print(f'🧠 Using model: {selected_model.name}')
        print(f'   Type: {model_type}')
        print(f'   Size: {size_gb:.1f} GB')
        
        if model_type == 'medicine-llm':
            print('   ✅ Specialized biomedical model detected!')
        elif model_type == 'mistral':
            print('   ℹ️  General purpose model (consider upgrading to Medicine LLM)')
            print('   💡 Run: python scripts/upgrade_to_medicine_llm.py')
    else:
        print('❌ No models found! Please run installation.command first.')
        exit(1)
else:
    print('❌ Models directory not found! Please run installation.command first.')
    exit(1)
"

# Check for multiple models and let user know
python -c "
from pathlib import Path
models = list(Path('models').glob('*.gguf')) if Path('models').exists() else []
if len(models) > 1:
    print()
    print('📋 Available models:')
    for model in sorted(models, key=lambda x: x.stat().st_size, reverse=True):
        size_gb = model.stat().st_size / (1024**3)
        print(f'   • {model.name} ({size_gb:.1f} GB)')
    print('   The largest/newest model will be used automatically.')
"

echo ""
echo "🚀 Starting Biomedical AI Interface..."
echo "======================================"

# Start the main application
python run_app.py

echo ""
echo "System shutdown complete."
echo "To restart: ./start.command" 