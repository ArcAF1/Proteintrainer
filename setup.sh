#!/usr/bin/env bash
set -euo pipefail

brew install cmake pkg-config libomp openssl@3 python@3.12 || true
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data_ingestion.py || true
python src/indexer.py || true
python src/gui.py
