# BioMedical AI Setup Instructions

## Step 1: Set HuggingFace Authentication

You need to provide your HuggingFace token for downloading the Mistral model.

### Option A: Environment Variable (Recommended)
```bash
export HUGGINGFACE_TOKEN="your-token-here"
```

### Option B: HuggingFace CLI
```bash
huggingface-cli login
```

## Step 2: Download Models

Once authenticated, run:
```bash
python scripts/download_models.py
```

This will:
1. Download Mistral-7B-Instruct-v0.2 (Q4_K_M quantization, ~4GB)
2. Configure M1-optimized inference settings
3. Install SpaCy language models
4. Save configuration for optimal performance

## Step 3: Verify Installation

After successful download, you should see:
- `models/mistral-7b-instruct.Q4_0.gguf` (~4GB)
- `models/m1_config.json` (M1 optimization settings)
- SpaCy models installed

## Step 4: Start the Application

```bash
./start.command
```

## Troubleshooting

If download fails:
1. Check your internet connection
2. Verify your HuggingFace token has access to the model
3. Try `huggingface-cli whoami` to verify authentication
4. Ensure you have at least 5GB free disk space

## M1 Performance Tips

The system is configured for optimal M1 performance:
- Metal GPU acceleration enabled
- All layers loaded to GPU
- Optimized batch sizes
- Half-precision computation where appropriate

Expected performance: 10-18 tokens/second on M1 MacBook 