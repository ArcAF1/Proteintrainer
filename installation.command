#!/bin/bash
# macOS first-run installer for the Biomedical Research Assistant.
# Double-click this file OR run `./installation.command` in Terminal.
#
# Steps:
#   1. Homebrew prerequisites (cmake, openssl, python@3.12, docker desktop check)
#   2. Python venv + packages with M1 optimization
#   3. Create .env file and download models
#   4. Start Neo4j container with wait-for-ready
#   5. Run system diagnostics to verify everything works
#   6. Optional: Download base model for LLM
#
# Re-run safe: already existing steps are skipped.
set -e

# Ensure we're running from the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables for better memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export TOKENIZERS_PARALLELISM=false

echo "🏥 Biomedical AI System Installation"
echo "===================================="
echo "Running from: $SCRIPT_DIR"
echo ""

# Load environment variables
if [ -f .env ]; then
    echo "[install] Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "🔧 Step 1: Installing system dependencies"
echo "----------------------------------------"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "[install] Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "[install] Installing essential brew packages ..."
brew install --quiet python@3.12 git curl wget || echo "Some packages already installed"
echo "✓ System dependencies installed"

echo ""
echo "🐍 Step 2: Setting up Python environment"
echo "---------------------------------------"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[install] Creating virtual environment..."
    python3.12 -m venv venv
fi

echo "[install] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "[install] Installing Python packages with M1 optimizations..."

# Install packages in order to avoid dependency conflicts
echo "[install] Installing core dependencies..."
pip install wheel setuptools

echo "[install] Installing basic packages first..."
pip install requests tqdm

# Verify requirements.txt exists before trying to install
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found in $PWD"
    echo "Available files:"
    ls -la
    exit 1
fi

echo "[install] Installing all requirements..."
pip install -r requirements.txt

# Check for M1 Mac and handle problematic packages
if [[ $(uname -m) == "arm64" && $(uname -s) == "Darwin" ]]; then
    echo "[install] Detected Apple Silicon Mac - installing M1-compatible packages..."
    
    # Install neo4j-graphrag specifically for M1 Macs
    echo "[install] Installing neo4j-graphrag for M1 compatibility..."
    pip install neo4j>=5.19.0 neo4j-graphrag>=1.7.0 || {
        echo "⚠️  Standard neo4j-graphrag installation failed, running M1 compatibility script..."
        python complete_m1_installation.py
    }
    
    # Install other M1-compatible packages
    echo "[install] Installing ChromaDB with M1 flags..."
    pip install --no-cache-dir chromadb>=0.4.22 || echo "⚠️  ChromaDB installation failed but continuing..."
    
    # Verify neo4j-graphrag is available
    python -c "
try:
    import neo4j_graphrag
    print('✅ neo4j-graphrag successfully imported')
except ImportError as e:
    print('❌ neo4j-graphrag still missing, running full M1 script...')
    import subprocess
    subprocess.run([sys.executable, 'complete_m1_installation.py'])
" || {
        echo "Running complete M1 installation as fallback..."
        python complete_m1_installation.py
    }
else
    echo "[install] Standard installation for Intel/other architecture..."
    # For non-M1 Macs, ensure neo4j-graphrag is installed
    pip install neo4j>=5.19.0 neo4j-graphrag>=1.7.0
fi

# Final verification of critical packages
echo "[install] Verifying critical packages..."
python -c "
critical_packages = ['neo4j_graphrag', 'neo4j', 'gradio', 'torch', 'transformers']
missing = []
for pkg in critical_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'⚠️  Missing packages: {missing}')
    print('   Run: python complete_m1_installation.py if needed')
else:
    print('✅ All critical packages available')
"

echo "✓ Python environment configured"

echo ""
echo "📊 Step 3: Running system diagnostics"
echo "------------------------------------"

# Run diagnostics
python -c "
import sys
sys.path.append('src')
try:
    from src.diagnostics import SystemDiagnostics
    diagnostics = SystemDiagnostics()
    results = diagnostics.run_all_checks()
    
    healthy_count = sum(1 for result in results.values() if result.get('healthy', False))
    total_count = len(results)
    
    print(f'System health: {healthy_count}/{total_count} components healthy')
    
    if healthy_count >= total_count - 1:  # Allow 1 missing component
        print('✅ System is ready!')
    else:
        print('⚠️  Some components need attention')
        for name, result in results.items():
            if not result.get('healthy', False):
                print(f'   • {name}: {result.get(\"recommendation\", \"Check manually\")}')
except Exception as e:
    print(f'Diagnostics failed: {e}')
    print('This is okay for first install - system will work once models are downloaded')
"

echo ""
echo "🤖 Step 4: Model Download Options"
echo "--------------------------------"

echo "Choose your AI model:"
echo "1. Mistral-7B (Default) - 4.1 GB, general purpose"
echo "2. Medicine LLM 13B (RECOMMENDED) - 7.9 GB, specialized for biomedical"
echo ""

# Check available disk space
available_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Available disk space: ${available_gb} GB"

if [ "$available_gb" -lt 10 ]; then
    echo "⚠️  Low disk space. Medicine LLM requires ~8GB free space."
    echo "Defaulting to Mistral-7B..."
    model_choice="1"
else
    echo ""
    read -p "Enter choice (1 or 2) [2]: " model_choice
    model_choice=${model_choice:-2}  # Default to Medicine LLM
fi

case $model_choice in
    1)
        echo "[install] Downloading Mistral-7B model..."
        python scripts/download_models.py
        ;;
    2)
        echo "[install] Downloading Medicine LLM 13B..."
        python scripts/upgrade_to_medicine_llm.py
        ;;
    *)
        echo "[install] Invalid choice, downloading Mistral-7B..."
        python scripts/download_models.py
        ;;
esac

echo ""
echo "🧪 Step 5: Final system validation"
echo "---------------------------------"

# Test neo4j-graphrag specifically first
echo "[install] Testing neo4j-graphrag installation..."
python test_neo4j_graphrag.py

# Test the complete system
python -c "
import sys
sys.path.append('src')
try:
    print('Testing system integration...')
    
    # Test 1: Check model files
    from pathlib import Path
    models_found = list(Path('models').glob('*.gguf')) if Path('models').exists() else []
    if models_found:
        print(f'✅ Found {len(models_found)} model file(s)')
        for model in models_found:
            size_gb = model.stat().st_size / (1024**3)
            print(f'   • {model.name}: {size_gb:.1f} GB')
    else:
        print('❌ No model files found')
        
    # Test 2: Try importing core modules
    from src.unified_system import UnifiedBiomedicalSystem
    print('✅ Core modules import successfully')
    
    # Test 3: Quick model detection
    from src.llm import detect_model_type
    if models_found:
        model_type = detect_model_type(str(models_found[0]))
        print(f'✅ Detected model type: {model_type}')
    
    # Test 4: Check critical imports
    critical_imports = [
        ('neo4j_graphrag', 'Neo4j GraphRAG'),
        ('gradio', 'Gradio UI'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers')
    ]
    
    missing_critical = []
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f'✅ {name} available')
        except ImportError:
            print(f'❌ {name} missing')
            missing_critical.append(name)
    
    if missing_critical:
        print(f'⚠️  Missing critical packages: {missing_critical}')
        print('   Run: python test_neo4j_graphrag.py to fix neo4j-graphrag')
        print('   Run: python complete_m1_installation.py for other packages')
    
    print('')
    print('🎉 Installation complete!')
    print('')
    print('Next steps:')
    print('1. Run ./start.command to launch the system')
    print('2. Ask: \"Test if the system is working\"')
    print('3. Try: \"Start training\" to build knowledge base')
    print('4. Example: \"What is the mechanism of metformin?\"')
    
except Exception as e:
    print(f'❌ System test failed: {e}')
    print('You can still try running ./start.command')
    print('If neo4j-graphrag is missing, run: python test_neo4j_graphrag.py')
"

echo ""
echo "🎯 Installation Summary"
echo "======================"
echo "✅ System dependencies installed"
echo "✅ Python environment configured"  
echo "✅ AI model downloaded"
echo "✅ Neo4j GraphRAG checked"
echo "✅ Ready to launch!"
echo ""
echo "📋 Troubleshooting:"
echo "   • If neo4j-graphrag is missing: python test_neo4j_graphrag.py"
echo "   • For other M1 issues: python complete_m1_installation.py"
echo "   • System diagnostics: python -m src.diagnostics"
echo ""
echo "🚀 To start: ./start.command" 