# Offline Medical RAG Assistant

This project provides a lightweight offline chat assistant for medical research running entirely on a Mac M1.

**Disclaimer:** This software is for research purposes only and does **not** constitute medical advice.

## Setup
1. Install [Homebrew](https://brew.sh/) if missing.
2. Install build tools and Python:
   ```bash
   brew install cmake pkg-config libomp openssl@3 python@3.12
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Download the LLM weights (mistral-7b-instruct Q4_0) and datasets listed in `src/data_sources.json`. DrugBank requires a free account; do not redistribute their XML.
5. Place models under `models/` and extracted text files under `data/`.
6. Optionally run `./setup.sh` to automate these steps.

## Building the Index
Adjust `chunk_size` and `top_k` in `src/config.py` if needed. Then run:
```bash
python src/indexer.py
```
This creates `indexes/pmc.faiss` and `pmc.pkl` for retrieval.

## Running the Chat UI
```bash
python src/gui.py
```
Open the local address printed in the terminal. The UI binds to `127.0.0.1` by default.

## Tests
```bash
pytest
```

## Data Sources
URLs for open datasets are provided in `src/data_sources.json`. Review their licenses before use. Run:
```bash
python src/data_ingestion.py
```
Indexing large datasets may consume >30 GB disk. Run `./uninstall.sh` to remove models and index files.
