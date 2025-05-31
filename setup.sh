#!/usr/bin/env bash
# Setup script for dependencies only - doesn't launch GUI

echo "[setup] Installing dependencies..."

# Install brew packages (ignore if already installed)
brew install cmake pkg-config libomp openssl@3 python@3.12 || true

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "[setup] Creating virtual environment..."
    python3.12 -m venv venv
fi

echo "[setup] Activating virtual environment..."
source venv/bin/activate

echo "[setup] Installing Python packages..."
pip install -r requirements.txt

echo "[setup] Installing SpaCy models..."
# First reinstall the specific versions we need
pip install markupsafe~=2.0 "packaging>=23.2,<24.0" --force-reinstall
# Then install SpaCy models
python scripts/install_spacy_models.py

echo "[setup] Setup complete!"
