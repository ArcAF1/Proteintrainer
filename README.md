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




<




