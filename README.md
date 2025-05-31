# Local Medical RAG Assistant 🩺

A 100% **locally-run** medical research assistant for Apple-Silicon Macs (M1/M2).  
No cloud API keys, no telemetry – all computation stays on your computer.

**Note**: While the AI runs locally, it can connect to the internet to fetch latest research papers and medical data from public sources (PubMed, arXiv, etc.).

## 🚀 Key Features

1. **Unified GUI** - Single, comprehensive interface combining all features
2. **Secure by Default** - Proper authentication with environment variables
3. **Progress Tracking** - Real-time progress indicators for all operations
4. **Graph Visualization** - Interactive Neo4j knowledge graph explorer
5. **Research Memory** - Persistent storage of findings with `/save` command
6. **Hypothesis Engine** - AI-powered research hypothesis generation
7. **Multi-source RAG** - FAISS vector search + Neo4j graph queries
8. **Local LLM** - Metal-accelerated Mistral 7B (no API keys needed)
9. **Internet Access** - Fetches latest research from PubMed, arXiv, DrugBank

## 📋 Main Capabilities

- Download open biomedical datasets (PubMed Central OA, DrugBank Open, ClinicalTrials.gov)
- Build a FAISS vector index **and** a Neo4j graph for verified entity relations
- Chat with locally-run 7B language model using retrieval-augmented generation (RAG)
- Graph visualization with slash commands (e.g., `/graph creatine`)
- Automated fetch of latest PubMed Central & arXiv papers on each training run
- Personal research memory with `/save` and `/recall` commands
- Agent mode with live PubMed/arXiv search (`/agent` command)
- GraphRAG mode for graph-aware answers (`/graphrag` command)

## 📚 Features

- **RAG Chat**: Query biomedical literature with citations and sources
- **Graph Explorer**: Interactive Neo4j knowledge graph visualization  
- **Hypothesis Generator**: AI-powered research hypothesis generation
- **Research Memory**: Persistent storage of research sessions and findings
- **Multi-Source Integration**: PubMed, PMC, arXiv, clinical trials, and more

## 📊 Data Sources

### Essential Datasets (10GB)
The system starts with three core datasets:

1. **ChEMBL** (4.6GB) - 2.4M+ bioactive compounds, drug targets, and bioassays
2. **Hetionet** (50MB) - Integrated network of biomedical knowledge with 47K nodes
3. **ClinicalTrials.gov** (5GB) - 450K+ clinical studies worldwide

### Additional Datasets (Available)
When you have more storage, you can add:
- PubMed/PMC articles (20-100GB+)
- PubChem compounds (300GB+) 
- UniProt proteins (100MB)
- STRING interactions (50MB)
- And 10+ more datasets

See [DATASETS.md](DATASETS.md) for complete documentation.

## 🏃 Quick Start (macOS 13+ Apple-Silicon)

### 1. Initial Setup
```bash
# Clone the repository
git clone <repo>
cd Proteintrainer

# Copy environment template and set secure password
cp env.example .env
# Edit .env to set your Neo4j password (IMPORTANT!)

# Install Docker Desktop for Mac if missing
# https://www.docker.com/products/docker-desktop/

# Start Neo4j database
docker-compose up -d
```

### 2. Launch Application
Double-click `start.command` in Finder, or run:
```bash
python run_app.py
```

The first run will:
- Create a Python 3.12 virtual environment
- Install all Python requirements  
- Start Neo4j Community Edition (ARM64) in Docker
- Open the unified GUI at http://localhost:7860

### 3. Initialize Data
1. Go to the **Setup** tab
2. Click **Test System** to verify all components
3. Click **Start Training** to download datasets and build indexes
   - Progress is shown in real-time
   - First run takes 10-30 minutes depending on internet speed
   - **Requires internet connection** to fetch medical datasets

### 4. Start Using
- **Chat tab**: Ask questions like "Vad är kreatin?" or "How does metformin work?"
- **Graph Explorer**: Visualize entity relationships
- **Research Lab**: Generate scientific hypotheses
- **Internet Search**: Use `/agent` to search latest papers online

## 🌐 Internet Connectivity

While the AI model runs entirely on your machine:
- **Training**: Downloads datasets from PubMed, DrugBank, ClinicalTrials.gov
- **Updates**: Fetches latest papers on each training run
- **Agent Mode**: Can search PubMed and arXiv in real-time
- **No Cloud AI**: All inference happens locally (no OpenAI, Anthropic, etc.)

## 🔐 Security Configuration

**IMPORTANT**: The system now uses proper authentication. You MUST:

1. Copy `env.example` to `.env`
2. Change the default Neo4j password in `.env`
3. Never commit `.env` to version control

Example `.env` file:
```bash
NEO4J_AUTH=neo4j/YourSecurePasswordHere123!
NEO4J_PASSWORD=YourSecurePasswordHere123!
```

## 💬 Chat Commands

- `/graph <entity>` - Visualize entity relationships
- `/save` - Save last Q&A to research memory
- `/recall <query>` - Search saved findings
- `/agent <query>` - Use agent with live web search
- `/graphrag <query>` - Use graph-aware RAG for complex queries

## 🛠️ Command-Line Usage

```bash
# One-off pipeline (same as GUI button)
python -m src.train_pipeline

# Run unified GUI
python src/gui_unified.py

# Run from main entry point
python run_app.py
```

## 📁 Project Structure

```text
├── start.command         # One-click launcher (macOS)
├── docker-compose.yml    # Neo4j Community container
├── env.example          # Environment template (copy to .env)
├── run_app.py          # Main entry point
├── src/                 # Application code
│   ├── gui_unified.py   # Unified GUI with all features
│   ├── train_pipeline.py # Training orchestrator
│   ├── data_ingestion.py # Dataset downloader with progress
│   └── ...
├── research_memory/     # Electronic lab notebook
├── data/               # Downloaded datasets
├── indexes/            # FAISS vector indexes
└── models/             # LLM weights (Mistral 7B)
```

## ⚡ Performance

- **Mac M1 16GB**: 10-18 tokens/second (llama-cpp-python Metal)
- **FAISS search**: < 50ms for 2M embeddings
- **Cypher queries**: < 1s for 3-hop graph traversals
- **RAM usage**: < 8GB with 7B model + indexes loaded

## 🔧 Troubleshooting

### Port Already In Use Error

If you see "Cannot find empty port in range: 7860-7860", the app is already running or didn't shut down properly.

**Solution 1**: Use the kill script
```bash
./kill_proteintrainer.sh
```

**Solution 2**: Manually kill the process
```bash
lsof -ti:7860 | xargs kill -9
```

**Solution 3**: The app will automatically try ports 7860-7865. Check console output for the actual port.

### Missing Models

If you get model loading errors, ensure the Mistral model is downloaded:
```bash
python -m src.download_models
```

## 🚨 Known Limitations

- Dataset URLs must be configured in `src/data_sources.json`
- Entity extraction uses regex fallback when SpaCy fails
- LoRA fine-tuning is CLI-only (GUI integration pending)
- GNN module is currently a stub implementation
- Requires internet for initial data download and updates

## 🤝 Contributing

Pull requests welcome! Please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation

## 📄 License

MIT - Use at your own risk. This is **NOT** medical advice.