#!/usr/bin/env python3
"""
CLI script for training biomedical LLMs

Usage:
    python train_biomedical.py --preset quick_test
    python train_biomedical.py --config configs/training_configs/overnight.yaml
    python train_biomedical.py --datasets hetionet,chembl_sqlite --epochs 2
"""
import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.biomedical_trainer import BiomedicalTrainer, TrainingConfig
from src.training_data_generator import BiomedicalDataGenerator


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_config_from_args(args) -> TrainingConfig:
    """Create training configuration from command line arguments."""
    return TrainingConfig(
        base_model=args.model,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_length,
        output_dir=args.output_dir or f"models/biomedical_cli"
    )


def main():
    parser = argparse.ArgumentParser(description="Train biomedical LLMs locally")
    
    # Preset or config file
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--preset", 
        choices=["quick_test", "overnight", "full_training"],
        help="Use predefined training preset"
    )
    group.add_argument(
        "--config", 
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Model settings
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model to fine-tune")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--grad-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    
    # LoRA settings
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Data settings
    parser.add_argument("--datasets", type=str, help="Comma-separated list of datasets (hetionet,chembl_sqlite,clinical_trials,pubmed)")
    parser.add_argument("--max-examples", type=int, default=10000, help="Maximum training examples")
    
    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory for model")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Flags
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--generate-data-only", action="store_true", help="Only generate training data")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.preset:
        config_file = f"configs/training_configs/{args.preset}.yaml"
        if not Path(config_file).exists():
            print(f"❌ Preset config file not found: {config_file}")
            sys.exit(1)
        config_dict = load_config_from_yaml(config_file)
        
        # Create config from YAML
        config = TrainingConfig(
            base_model=config_dict.get("base_model", args.model),
            num_epochs=config_dict.get("num_epochs", args.epochs),
            max_steps=config_dict.get("max_steps", args.max_steps),
            per_device_train_batch_size=config_dict.get("per_device_train_batch_size", args.batch_size),
            gradient_accumulation_steps=config_dict.get("gradient_accumulation_steps", args.grad_accumulation),
            learning_rate=config_dict.get("learning_rate", args.learning_rate),
            lora_r=config_dict.get("lora_r", args.lora_r),
            lora_alpha=config_dict.get("lora_alpha", args.lora_alpha),
            lora_dropout=config_dict.get("lora_dropout", args.lora_dropout),
            max_seq_length=config_dict.get("max_seq_length", args.max_length),
            load_in_4bit=config_dict.get("load_in_4bit", not args.no_4bit),
            output_dir=config_dict.get("output_dir", args.output_dir or f"models/{args.preset}")
        )
        
        selected_datasets = config_dict.get("selected_datasets", ["hetionet"])
        max_examples = config_dict.get("max_examples", args.max_examples)
        
    elif args.config:
        if not Path(args.config).exists():
            print(f"❌ Config file not found: {args.config}")
            sys.exit(1)
        config_dict = load_config_from_yaml(args.config)
        # Similar config creation as above
        config = create_config_from_args(args)  # Simplified for now
        selected_datasets = config_dict.get("selected_datasets", ["hetionet"])
        max_examples = config_dict.get("max_examples", args.max_examples)
        
    else:
        # Use command line arguments
        config = create_config_from_args(args)
        selected_datasets = args.datasets.split(",") if args.datasets else ["hetionet"]
        max_examples = args.max_examples
    
    print(f"🚀 Starting biomedical LLM training...")
    print(f"📊 Configuration:")
    print(f"   Model: {config.base_model}")
    print(f"   Datasets: {selected_datasets}")
    print(f"   Max examples: {max_examples}")
    print(f"   Output: {config.output_dir}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Max steps: {config.max_steps}")
    
    # Initialize trainer
    trainer = BiomedicalTrainer(config)
    
    try:
        if args.generate_data_only:
            print("🔄 Generating training data only...")
            data_files = trainer.generate_training_data(max_examples)
            print(f"✅ Training data saved to: {trainer.output_dir / 'training_data'}")
            for split, file_path in data_files.items():
                print(f"   {split}: {file_path}")
            
        elif args.eval_only:
            print("📊 Running evaluation only...")
            if not (Path(config.output_dir) / "pytorch_model.bin").exists():
                print("❌ No trained model found. Train first.")
                sys.exit(1)
            results = trainer.evaluate_model()
            print(f"✅ Evaluation complete: {results}")
            
        else:
            print("🧠 Starting full training...")
            trainer.train(resume_from_checkpoint=args.resume)
            
            # Run evaluation
            print("📊 Running post-training evaluation...")
            trainer.evaluate_model()
            
            # Generate sample outputs
            print("🔬 Testing sample outputs...")
            test_prompts = [
                "What is the mechanism of action of aspirin?",
                "Explain the relationship between diabetes and cardiovascular disease.",
                "What are the side effects of metformin?"
            ]
            
            outputs = trainer.generate_sample_outputs(test_prompts)
            print("\n📝 Sample outputs:")
            for prompt, output in zip(test_prompts, outputs):
                print(f"\nQ: {prompt}")
                print(f"A: {output}")
                print("-" * 50)
                
        print("\n🎉 Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 