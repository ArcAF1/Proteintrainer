#!/bin/bash

echo "=========================================="
echo "🚀 Proteintrainer Full Training"
echo "=========================================="
echo ""
echo "This will download large datasets:"
echo "- PubMed Central (several GB)"
echo "- DrugBank database"
echo "- ClinicalTrials.gov data"
echo ""
echo "Make sure you have:"
echo "- Stable internet connection"
echo "- At least 10GB free disk space"
echo "- Time (this may take 30-60 minutes)"
echo ""
read -p "Continue with full training? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Navigate to the script's directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Ensure Neo4j is running
if ! docker ps | grep -q medical-neo4j; then
    echo "Starting Neo4j..."
    docker-compose up -d
    sleep 10
fi

# Run the training pipeline with large downloads enabled
echo ""
echo "📥 Starting full dataset download and training..."
echo ""

# Make sure SKIP_LARGE_DOWNLOADS is not set
unset SKIP_LARGE_DOWNLOADS

# Run the training pipeline
python -m src.train_pipeline

echo ""
echo "✅ Full training complete!"
echo ""
echo "You now have:"
echo "- Full PubMed Central articles indexed"
echo "- Complete DrugBank database"
echo "- All clinical trials data"
echo ""
echo "Run ./start.command to use the system with full knowledge base!" 