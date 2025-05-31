# Developer README – Cursor edition

## Local dev setup
```bash
# Prerequisites
brew install python@3.12 cmake pkg-config libomp openssl@3
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# optional: run Neo4j container for tests
docker compose up -d
```

## Run tests
```bash
pytest -q
```
All tests are mocked to avoid large downloads and finish in < 60 s on an M1.

## Directory hints
* `src/` – main code base
  * `train_pipeline.py` – end-to-end ingestion/index/graph orchestration
  * `rag_chat.py` – RAG pipeline used by GUI
  * `fine_tune.py` – LoRA adapter script (lightweight)
* `research_memory/` – electronic-lab-notebook helper layer (MIT-licensed)
* `data/parsed/` – place `articles.json` & `relations.json` if you generate your own entities.

## Extending
1. **Entity extraction** – integrate a NER model (e.g. SciSpacy) during ingestion and dump `articles.json` / `relations.json`, then retrain via GUI.
2. **Citations** – modify `rag_chat.RAGChat.retrieve()` to keep track of document ids and format answers like `[1]`.
3. **Save chat** – add a `/save` command that appends the current chat history to ResearchMemory.
4. **Fine-tune hook in GUI** – create another Gradio button that calls `fine_tune.py` with user-provided jsonl.

## Known issues / TODO
| # | Item |
|---|------|
|1| `data_sources.json` still contains placeholder links |
|2| Citation formatting not yet `[1]`, `[2]` style | ~~resolved~~ |
|3| NER + relation extraction not implemented | ~~resolved (naïve extractor)~~ |
|4| Tests do not cover Neo4j graph build |
|5| GUI lacks `/save` & fine-tune button | `/save` resolved (fine-tune pending)`

Pull requests welcome. 