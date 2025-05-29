# Offline Medical RAG Assistant

This project provides a lightweight offline chat assistant for medical research. It uses open-source models and datasets to run entirely on a MacBook M1 with 16 GB RAM.

**Disclaimer:** This software is for research purposes only and does **not** constitute medical advice.

## Setup
1. Install [Homebrew](https://brew.sh/) if missing.
2. Install Python 3.12 via Homebrew:
   ```bash
   brew install python@3.12
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Download sentence-transformer and LLM weights manually (see `data_sources.json`).
5. Place downloaded models under `models/` and datasets under `data/`.

## Building the Index
Run the indexer after datasets are prepared:
```bash
python src/indexer.py
```
This generates `indexes/pmc.faiss` used for retrieval.

## Running the Chat UI
Start the Gradio interface:
```bash
python src/gui.py
```
Open the local address printed in the terminal. Chat responses will cite retrieved text passages.

## Tests
Run the basic pipeline test:
```bash
pytest
```

## Data Sources
Dataset download links are placeholders in `src/data_sources.json`. Fill them in and execute:
```bash
python src/data_ingestion.py
```
This may take a long time and requires substantial disk space.

## Plan for Building a Local AI Chat Assistant on Mac M1
The following notes outline how a non-technical user can build and run a
fully local chat assistant. They serve as general guidance rather than
strict rules.

1. **Choose a lightweight model.** Mistral 7B Instruct (4‑bit) or LLaMA 2 7B
   are good options that fit within 16 GB RAM using [Ollama](https://ollama.com).
2. **Install helper tools.** Use Homebrew to install Python 3.12 and Ollama.
3. **Download open medical datasets** such as PubMed Central OA, DrugBank Open
   Data, and ClinicalTrials.gov. Fill URLs in `src/data_sources.json` and run
   `python src/data_ingestion.py` to prepare the files locally.
4. **Build a FAISS index** with `python src/indexer.py`. This step may take
   several hours depending on dataset size.
5. **Run the chat UI** via `python src/gui.py` and open the printed address in
   your browser. Ask questions in Swedish and the assistant will reference
   retrieved documents as “(Dok n)”.

The entire workflow is offline after the initial downloads and keeps within a
budget well below 5000 SEK.


