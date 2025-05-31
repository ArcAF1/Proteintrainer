"""
Complete Biomedical LLM Training System for Mac M1

Features:
- QLoRA 4-bit quantization training
- M1/Metal optimization with MPS backend
- Memory management and gradient checkpointing
- Real-time monitoring and evaluation
- Medical safety validation
- Resumable training with checkpoints
"""
from __future__ import annotations

import json
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import gc
import psutil
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
import evaluate

from .config import settings
from .training_data_generator import BiomedicalDataGenerator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for biomedical LLM."""
    
    # Model settings
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 100
    max_steps: int = -1
    
    # Memory optimization
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False  # False for M1
    fp16: bool = False  # Use bf16 on M1
    bf16: bool = True
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    
    # Output paths
    output_dir: str = "models/biomedical_lora"
    logging_dir: str = "logs/tensorboard"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.memory_history = []
        self.gpu_memory_history = []
        
    def log_memory(self, step: int, prefix: str = ""):
        """Log current memory usage."""
        # System memory
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        
        # MPS memory (if available)
        mps_memory_gb = 0
        if torch.backends.mps.is_available():
            mps_memory_gb = torch.mps.current_allocated_memory() / (1024**3)
            
        self.memory_history.append({
            'step': step,
            'system_memory_gb': memory_gb,
            'mps_memory_gb': mps_memory_gb,
            'prefix': prefix
        })
        
        logger.info(f"{prefix} Step {step}: System Memory: {memory_gb:.2f}GB, "
                   f"MPS Memory: {mps_memory_gb:.2f}GB")
        
    def clear_memory(self):
        """Force garbage collection and clear MPS cache."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Cleared memory caches")
        
    def plot_memory_usage(self, save_path: str):
        """Plot memory usage over time."""
        if not self.memory_history:
            return
            
        steps = [m['step'] for m in self.memory_history]
        system_memory = [m['system_memory_gb'] for m in self.memory_history]
        mps_memory = [m['mps_memory_gb'] for m in self.memory_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, system_memory, label='System Memory (GB)', linewidth=2)
        plt.plot(steps, mps_memory, label='MPS Memory (GB)', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Memory Usage During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Memory usage plot saved to {save_path}")


class MedicalSafetyValidator:
    """Validate model outputs for medical safety."""
    
    def __init__(self):
        self.safety_keywords = [
            "self-medicate", "ignore doctor", "stop medication",
            "replace prescription", "cure cancer", "diagnose yourself"
        ]
        
    def validate_output(self, text: str) -> Tuple[bool, float, str]:
        """
        Validate if output is medically safe.
        
        Returns:
            (is_safe, confidence_score, explanation)
        """
        text_lower = text.lower()
        
        # Check for dangerous advice
        for keyword in self.safety_keywords:
            if keyword in text_lower:
                return False, 0.9, f"Contains dangerous advice: '{keyword}'"
                
        # Check for appropriate disclaimers
        has_disclaimer = any(phrase in text_lower for phrase in [
            "consult", "healthcare professional", "medical advice",
            "doctor", "physician", "not medical advice"
        ])
        
        if not has_disclaimer and any(word in text_lower for word in [
            "treat", "cure", "medicine", "drug", "dose"
        ]):
            return False, 0.7, "Medical advice without proper disclaimer"
            
        return True, 0.9, "Safe medical content"


class BiomedicalTrainerCallback(TrainerCallback):
    """Custom callback for biomedical training monitoring."""
    
    def __init__(self, memory_monitor: MemoryMonitor, safety_validator: MedicalSafetyValidator):
        self.memory_monitor = memory_monitor
        self.safety_validator = safety_validator
        self.start_time = time.time()
        
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            self.memory_monitor.log_memory(state.global_step, "Training")
            
    def on_evaluate(self, args, state, control, **kwargs):
        self.memory_monitor.log_memory(state.global_step, "Evaluation")
        
    def on_train_end(self, args, state, control, **kwargs):
        training_time = time.time() - self.start_time
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        
        # Plot final memory usage
        plot_path = Path(args.output_dir) / "memory_usage.png"
        self.memory_monitor.plot_memory_usage(str(plot_path))


class BiomedicalDataset(Dataset):
    """Dataset for biomedical instruction tuning."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                self.data = [json.loads(line) for line in f]
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
                
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Format as instruction-following
        if 'input' in example and example['input']:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
            
        text = prompt + example['output']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class BiomedicalTrainer:
    """Complete biomedical LLM training system."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.safety_validator = MedicalSafetyValidator()
        
        # Setup output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Setup device (MPS for M1)
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")
        
    def _setup_device(self) -> str:
        """Setup optimal device for M1 Mac."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with M1 optimizations."""
        logger.info(f"Loading model: {self.config.base_model}")
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Setup quantization config for 4-bit training
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ) if self.config.load_in_4bit else None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            device_map="auto" if self.device != "mps" else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to MPS if needed
        if self.device == "mps" and quantization_config is None:
            self.model = self.model.to(self.device)
            
        # Prepare for k-bit training
        if self.config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        logger.info("Model and tokenizer setup complete")
        
    def setup_lora(self):
        """Setup LoRA configuration."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup complete")
        
    def generate_training_data(self, max_examples: int = 25000) -> Dict[str, str]:
        """Generate training data from biomedical sources."""
        logger.info("Generating training data...")
        
        generator = BiomedicalDataGenerator(self.output_dir / "training_data")
        generator.generate_all(max_examples=max_examples)
        generator.add_data_augmentation()
        files = generator.save_training_data()
        
        return files
        
    def prepare_datasets(self, data_files: Dict[str, str]):
        """Prepare training and validation datasets."""
        train_dataset = BiomedicalDataset(
            data_files['train'], 
            self.tokenizer, 
            self.config.max_seq_length
        )
        
        eval_dataset = BiomedicalDataset(
            data_files['validation'], 
            self.tokenizer, 
            self.config.max_seq_length
        )
        
        return train_dataset, eval_dataset
    
    def setup_training_args(self):
        """Setup training arguments optimized for M1."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Optimizer settings
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            
            # Precision settings for M1
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            
            # Evaluation and logging
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            
            # Memory optimization
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            gradient_checkpointing=self.config.gradient_checkpointing,
            
            # Output settings
            logging_dir=self.config.logging_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Disable unwanted features for M1
            use_mps_device=(self.device == "mps"),
            remove_unused_columns=False,
            report_to=["tensorboard", "wandb"] if wandb.run else ["tensorboard"],
        )
        
        return training_args
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Safety validation
        safety_scores = []
        for pred in decoded_preds[:10]:  # Sample for validation
            is_safe, score, _ = self.safety_validator.validate_output(pred)
            safety_scores.append(score if is_safe else 0.0)
        
        return {
            "safety_score": np.mean(safety_scores) if safety_scores else 0.0,
            "avg_pred_length": np.mean([len(pred.split()) for pred in decoded_preds])
        }
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop."""
        logger.info("Starting biomedical LLM training...")
        self.memory_monitor.log_memory(0, "Pre-training")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        self.setup_lora()
        
        # Generate or load training data
        data_files = self.generate_training_data()
        train_dataset, eval_dataset = self.prepare_datasets(data_files)
        
        # Setup training
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[BiomedicalTrainerCallback(self.memory_monitor, self.safety_validator)]
        )
        
        # Start training
        try:
            if resume_from_checkpoint:
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
                self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                self.trainer.train()
                
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.memory_monitor.clear_memory()
    
    def evaluate_model(self, test_data_path: Optional[str] = None):
        """Evaluate trained model."""
        if not self.trainer:
            raise ValueError("Model not trained yet")
            
        logger.info("Evaluating model...")
        
        if test_data_path:
            test_dataset = BiomedicalDataset(test_data_path, self.tokenizer)
            results = self.trainer.evaluate(test_dataset)
        else:
            results = self.trainer.evaluate()
            
        logger.info(f"Evaluation results: {results}")
        
        # Save evaluation results
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def generate_sample_outputs(self, prompts: List[str]) -> List[str]:
        """Generate sample outputs for evaluation."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
            
        outputs = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append(output[len(prompt):].strip())
            
        return outputs


def main():
    """Main training function."""
    # Setup configuration
    config = TrainingConfig(
        base_model="mistralai/Mistral-7B-Instruct-v0.2",
        num_epochs=2,
        max_steps=1000,  # Quick training for testing
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        output_dir="models/biomedical_mistral_lora"
    )
    
    # Initialize trainer
    trainer = BiomedicalTrainer(config)
    
    # Start training
    trainer.train()
    
    # Evaluate
    trainer.evaluate_model()
    
    # Test sample outputs
    test_prompts = [
        "What is the mechanism of action of aspirin?",
        "Explain the relationship between diabetes and cardiovascular disease.",
        "What are the side effects of metformin?"
    ]
    
    outputs = trainer.generate_sample_outputs(test_prompts)
    for prompt, output in zip(test_prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print("-" * 50)


if __name__ == "__main__":
    main() 