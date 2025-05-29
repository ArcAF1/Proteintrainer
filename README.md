# Offline Medical RAG Assistant

This project provides a lightweight offline chat assistant for medical research. It uses open-source models and datasets to run entirely on a MacBook M1.

**Disclaimer:** This software is for research purposes only and does **not** constitute medical advice.

## Setup
1. Install [Homebrew](https://brew.sh/) if missing.
2. Install Python 3.12:
   ```bash
   brew install python@3.12
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Download the sentence-transformer model and LLM weights manually (see `src/data_sources.json`). Place them under `models/` and datasets under `data/`.

## Building the Index
```bash
python src/indexer.py
```
This creates `indexes/pmc.faiss` used for retrieval.

## Running the Chat UI
```bash
python src/gui.py
```
Open the local address printed in the terminal.

## Research Memory
An optional electronic lab notebook is provided in `research_memory/`.
Install it in editable mode:
```bash
pip install -e research_memory
```
Use the `researchmem` CLI to create and search notes.

## Tests
```bash
pytest
```
