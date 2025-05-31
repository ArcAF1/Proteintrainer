"""Training connector that bridges the GUI with the biomedical trainer system."""

import time
import traceback
from typing import Optional, Callable
from pathlib import Path
import platform

from .biomedical_trainer import BiomedicalTrainer, TrainingConfig


class TrainingConnector:
    """Connects GUI to training system with progress tracking."""
    
    def __init__(self):
        self.trainer: Optional[BiomedicalTrainer] = None
        self.is_training = False
        
    def start_full_training(self, progress_callback: Optional[Callable[[float, str], None]] = None):
        """Start full biomedical LLM training with progress updates."""
        
        if self.is_training:
            raise RuntimeError("Training already in progress")
        
        try:
            self.is_training = True
            
            if progress_callback:
                progress_callback(0, "Initializing training configuration...")
            
            # Configure training for M1 Mac with reasonable settings
            config = TrainingConfig(
                base_model="mistralai/Mistral-7B-Instruct-v0.2",
                num_epochs=1,  # Start with 1 epoch
                max_steps=500,  # Limit steps for reasonable completion time
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,  # Effective batch size of 8
                learning_rate=2e-4,
                output_dir="models/biomedical_mistral_lora",
                
                # M1 optimizations
                fp16=False,  # Use bf16 instead
                bf16=True,
                gradient_checkpointing=True,
                dataloader_pin_memory=False,
                
                # More frequent evaluation for progress tracking
                evaluation_strategy="steps",
                eval_steps=50,
                save_steps=100,
                logging_steps=10
            )
            
            if progress_callback:
                progress_callback(5, "Setting up biomedical trainer...")
            
            # Initialize trainer
            self.trainer = BiomedicalTrainer(config)
            
            if progress_callback:
                progress_callback(10, "Loading model and tokenizer...")
            
            # Setup model components
            self.trainer.setup_model_and_tokenizer()
            
            if progress_callback:
                progress_callback(20, "Configuring LoRA adapters...")
                
            self.trainer.setup_lora()
            
            if progress_callback:
                progress_callback(30, "Generating training data...")
            
            # Generate training data
            data_files = self.trainer.generate_training_data(max_examples=5000)  # Smaller dataset for faster training
            
            if progress_callback:
                progress_callback(40, "Preparing datasets...")
            
            train_dataset, eval_dataset = self.trainer.prepare_datasets(data_files)
            
            if progress_callback:
                progress_callback(50, "Starting fine-tuning...")
            
            # Custom training loop with progress updates
            training_args = self.trainer.setup_training_args()
            
            from transformers import DataCollatorForLanguageModeling, Trainer
            from .biomedical_trainer import BiomedicalTrainerCallback
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.trainer.tokenizer,
                mlm=False
            )
            
            # Custom callback for progress tracking
            class ProgressCallback:
                def __init__(self, progress_fn, total_steps):
                    self.progress_fn = progress_fn
                    self.total_steps = total_steps
                    self.start_progress = 50  # We're already at 50%
                    
                def on_step_end(self, args, state, control, **kwargs):
                    # Throttle to every 5 steps to avoid flooding the GUI
                    if state.global_step % 5 != 0:
                        return
                    if self.progress_fn and self.total_steps > 0:
                        step_progress = (state.global_step / self.total_steps) * 40  # Remaining 40%
                        total_progress = self.start_progress + step_progress

                        # Get loss if available
                        loss_info = ""
                        if hasattr(state, 'log_history') and state.log_history:
                            recent_logs = [log for log in state.log_history if 'train_loss' in log]
                            if recent_logs:
                                loss_info = f" (loss: {recent_logs[-1]['train_loss']:.4f})"

                        self.progress_fn(
                            total_progress,
                            f"Fine-tuning step {state.global_step}/{self.total_steps}{loss_info}"
                        )
            
            # Calculate total steps
            total_steps = len(train_dataset) // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
            if config.max_steps > 0:
                total_steps = min(total_steps, config.max_steps)
            
            # Setup trainer with progress callback
            callbacks = [BiomedicalTrainerCallback(self.trainer.memory_monitor, self.trainer.safety_validator)]
            if progress_callback:
                callbacks.append(ProgressCallback(progress_callback, total_steps))
            
            trainer = Trainer(
                model=self.trainer.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self.trainer.compute_metrics,
                callbacks=callbacks
            )
            
            self.trainer.trainer = trainer
            
            # Start training
            trainer.train()
            
            if progress_callback:
                progress_callback(90, "Saving trained model...")
            
            # Save final model
            trainer.save_model()
            self.trainer.tokenizer.save_pretrained(self.trainer.output_dir)
            
            if progress_callback:
                progress_callback(95, "Running final evaluation...")
            
            # Quick evaluation
            try:
                eval_results = trainer.evaluate()
                if progress_callback:
                    progress_callback(100, f"Training complete! Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
            except Exception as e:
                if progress_callback:
                    progress_callback(100, "Training complete! (evaluation skipped)")
            
            # Test a few sample outputs
            if progress_callback:
                progress_callback(100, "Training complete! Testing sample outputs...")
            
            test_prompts = [
                "What is the mechanism of action of metformin?",
                "Explain how insulin works in the body.",
                "What are the benefits of creatine supplementation?"
            ]
            
            try:
                outputs = self.trainer.generate_sample_outputs(test_prompts)
                print("\n🧪 **Sample Model Outputs:**")
                for prompt, output in zip(test_prompts, outputs):
                    print(f"\nQ: {prompt}")
                    print(f"A: {output[:200]}...")
            except Exception as e:
                print(f"Sample generation failed: {e}")
            
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # Apple Silicon: run only data pipeline
                from . import train_pipeline

                def pipe_cb(p, m):
                    if progress_callback:
                        progress_callback(p * 100, m)

                if progress_callback:
                    progress_callback(0, "Running data pipeline (fine-tune skipped on Apple Silicon)…")

                train_pipeline.main(progress_callback=pipe_cb)

                if progress_callback:
                    progress_callback(100, "✅ Data pipeline complete! Fine-tuning skipped on Apple Silicon.")
                print("[TrainingConnector] Finished data pipeline; skipped LoRA fine-tune on Apple Silicon.")
                return True
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"Training error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            
            if progress_callback:
                progress_callback(0, f"Training failed: {error_msg}")
            
            raise RuntimeError(f"Training failed: {error_msg}")
            
        finally:
            self.is_training = False
    
    def get_training_status(self) -> dict:
        """Get current training status and model info."""
        model_dir = Path("models/biomedical_mistral_lora")
        
        status = {
            "is_training": self.is_training,
            "has_trained_model": False,
            "model_size": 0,
            "last_training": None
        }
        
        if model_dir.exists():
            # Check for trained model files
            adapter_files = list(model_dir.glob("adapter_*.safetensors"))
            config_file = model_dir / "adapter_config.json"
            
            if adapter_files and config_file.exists():
                status["has_trained_model"] = True
                status["model_size"] = sum(f.stat().st_size for f in adapter_files) / (1024 * 1024)  # MB
                status["last_training"] = max(f.stat().st_mtime for f in adapter_files)
        
        return status
    
    def load_trained_model(self):
        """Load a previously trained model for inference."""
        if self.trainer:
            return self.trainer
        
        model_dir = Path("models/biomedical_mistral_lora")
        if not model_dir.exists():
            raise FileNotFoundError("No trained model found")
        
        # Load the trained model
        config = TrainingConfig(output_dir=str(model_dir))
        self.trainer = BiomedicalTrainer(config)
        
        # Load model components
        self.trainer.setup_model_and_tokenizer()
        
        # Load LoRA weights
        from peft import PeftModel
        self.trainer.model = PeftModel.from_pretrained(
            self.trainer.model, 
            str(model_dir)
        )
        
        return self.trainer


# Global instance
_training_connector: Optional[TrainingConnector] = None


def get_training_connector() -> TrainingConnector:
    """Get the global training connector instance."""
    global _training_connector
    if _training_connector is None:
        _training_connector = TrainingConnector()
    return _training_connector 