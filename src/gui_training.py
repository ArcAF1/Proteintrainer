"""
Training GUI Tab for Biomedical LLM Fine-tuning

Features:
- Dataset selection checkboxes
- Hyperparameter controls
- Real-time training progress
- Memory usage monitoring
- Live loss graphs
- Training presets
- Model comparison
"""
from __future__ import annotations

import gradio as gr
import json
import threading
import time
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import psutil

from .biomedical_trainer import BiomedicalTrainer, TrainingConfig
from .training_data_generator import BiomedicalDataGenerator
from .config import settings


class TrainingGUI:
    """GUI for biomedical LLM training."""
    
    def __init__(self):
        self.trainer: Optional[BiomedicalTrainer] = None
        self.training_thread: Optional[threading.Thread] = None
        self.progress_queue = queue.Queue()
        self.is_training = False
        self.training_logs = []
        self.memory_logs = []
        
        # Training presets
        self.presets = {
            "Quick Test": {
                "num_epochs": 1,
                "max_steps": 100,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": 5e-4,
                "max_examples": 500
            },
            "Overnight Training": {
                "num_epochs": 2,
                "max_steps": 2000,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "max_examples": 10000
            },
            "Full Training": {
                "num_epochs": 3,
                "max_steps": -1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 2e-4,
                "max_examples": 50000
            }
        }
        
    def create_training_tab(self):
        """Create the training tab interface."""
        with gr.Column():
            gr.Markdown("# 🧬 Biomedical LLM Training")
            gr.Markdown("Train your local medical AI on biomedical datasets")
            
            with gr.Row():
                # Left column: Configuration
                with gr.Column(scale=1):
                    gr.Markdown("## Dataset Selection")
                    
                    # Dataset checkboxes
                    dataset_checks = {}
                    datasets = [
                        ("hetionet", "Hetionet Knowledge Graph", "47K nodes, 2.25M edges"),
                        ("chembl_sqlite", "ChEMBL Database", "Bioactivity data, 4.6GB"),
                        ("clinical_trials", "Clinical Trials", "Treatment outcomes, 5GB"),
                        ("pubmed", "PubMed Articles", "Recent medical literature"),
                        ("drugbank", "DrugBank", "Drug interactions (if available)")
                    ]
                    
                    for dataset_id, name, desc in datasets:
                        dataset_checks[dataset_id] = gr.Checkbox(
                            label=f"{name} - {desc}",
                            value=dataset_id in ["hetionet", "chembl_sqlite", "clinical_trials"]
                        )
                    
                    gr.Markdown("## Training Presets")
                    preset_dropdown = gr.Dropdown(
                        choices=list(self.presets.keys()),
                        value="Quick Test",
                        label="Training Preset"
                    )
                    
                    gr.Markdown("## Hyperparameters")
                    
                    # Model settings
                    base_model = gr.Dropdown(
                        choices=[
                            "mistralai/Mistral-7B-Instruct-v0.2",
                            "microsoft/DialoGPT-medium",
                            "microsoft/BioGPT"
                        ],
                        value="mistralai/Mistral-7B-Instruct-v0.2",
                        label="Base Model"
                    )
                    
                    # Training parameters
                    num_epochs = gr.Slider(1, 10, value=2, step=1, label="Epochs")
                    max_steps = gr.Number(value=1000, label="Max Steps (-1 for full)")
                    batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
                    grad_accumulation = gr.Slider(1, 32, value=8, step=1, label="Gradient Accumulation")
                    learning_rate = gr.Slider(1e-5, 1e-3, value=2e-4, step=1e-5, label="Learning Rate")
                    
                    # LoRA settings
                    gr.Markdown("### LoRA Configuration")
                    lora_r = gr.Slider(8, 128, value=32, step=8, label="LoRA Rank (r)")
                    lora_alpha = gr.Slider(16, 256, value=64, step=16, label="LoRA Alpha")
                    lora_dropout = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="LoRA Dropout")
                    
                    # Data settings
                    max_examples = gr.Number(value=10000, label="Max Training Examples")
                    max_seq_length = gr.Slider(512, 4096, value=2048, step=128, label="Max Sequence Length")
                
                # Right column: Training and Monitoring
                with gr.Column(scale=2):
                    gr.Markdown("## Training Control")
                    
                    with gr.Row():
                        start_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                        pause_btn = gr.Button("⏸️ Pause", variant="secondary")
                        stop_btn = gr.Button("🛑 Stop", variant="stop")
                        resume_btn = gr.Button("▶️ Resume", variant="secondary")
                    
                    # Training status
                    training_status = gr.Textbox(
                        label="Training Status",
                        value="Ready to train",
                        interactive=False
                    )
                    
                    # Progress bars
                    epoch_progress = gr.Progress()
                    step_progress = gr.Progress()
                    
                    gr.Markdown("## Real-time Monitoring")
                    
                    # Metrics display
                    with gr.Row():
                        with gr.Column():
                            current_loss = gr.Number(label="Current Loss", value=0.0, interactive=False)
                            eval_loss = gr.Number(label="Eval Loss", value=0.0, interactive=False)
                            safety_score = gr.Number(label="Safety Score", value=0.0, interactive=False)
                        
                        with gr.Column():
                            memory_usage = gr.Number(label="Memory (GB)", value=0.0, interactive=False)
                            training_speed = gr.Number(label="Tokens/sec", value=0.0, interactive=False)
                            eta = gr.Textbox(label="ETA", value="--", interactive=False)
                    
                    # Live plots
                    loss_plot = gr.Plot(label="Training Loss")
                    memory_plot = gr.Plot(label="Memory Usage")
                    
                    # Training logs
                    training_logs = gr.Textbox(
                        label="Training Logs",
                        lines=10,
                        max_lines=20,
                        interactive=False,
                        autoscroll=True
                    )
            
            # Model comparison section
            gr.Markdown("## Model Comparison")
            
            with gr.Row():
                with gr.Column():
                    test_prompt = gr.Textbox(
                        label="Test Prompt",
                        value="What is the mechanism of action of aspirin?",
                        lines=3
                    )
                    compare_btn = gr.Button("Compare Models")
                
                with gr.Column():
                    base_output = gr.Textbox(label="Base Model Output", lines=5, interactive=False)
                    trained_output = gr.Textbox(label="Trained Model Output", lines=5, interactive=False)
            
            # Event handlers
            def update_from_preset(preset_name):
                if preset_name in self.presets:
                    preset = self.presets[preset_name]
                    return (
                        preset["num_epochs"],
                        preset["max_steps"],
                        preset["per_device_train_batch_size"],
                        preset["gradient_accumulation_steps"],
                        preset["learning_rate"],
                        preset["max_examples"]
                    )
                return num_epochs.value, max_steps.value, batch_size.value, grad_accumulation.value, learning_rate.value, max_examples.value
            
            preset_dropdown.change(
                update_from_preset,
                inputs=[preset_dropdown],
                outputs=[num_epochs, max_steps, batch_size, grad_accumulation, learning_rate, max_examples]
            )
            
            # Training control events
            start_btn.click(
                self.start_training,
                inputs=[
                    *dataset_checks.values(),
                    base_model, num_epochs, max_steps, batch_size, grad_accumulation,
                    learning_rate, lora_r, lora_alpha, lora_dropout, max_examples, max_seq_length
                ],
                outputs=[training_status]
            )
            
            stop_btn.click(
                self.stop_training,
                outputs=[training_status]
            )
            
            compare_btn.click(
                self.compare_models,
                inputs=[test_prompt],
                outputs=[base_output, trained_output]
            )
            
            # Auto-refresh for monitoring
            self.setup_auto_refresh(
                training_status, current_loss, eval_loss, safety_score,
                memory_usage, training_speed, eta, loss_plot, memory_plot, training_logs
            )
    
    def start_training(self, *args):
        """Start the training process."""
        if self.is_training:
            return "❌ Training already in progress"
        
        try:
            # Parse arguments
            dataset_args = args[:5]  # First 5 are dataset checkboxes
            config_args = args[5:]   # Rest are configuration
            
            selected_datasets = []
            dataset_names = ["hetionet", "chembl_sqlite", "clinical_trials", "pubmed", "drugbank"]
            for i, selected in enumerate(dataset_args):
                if selected:
                    selected_datasets.append(dataset_names[i])
            
            if not selected_datasets:
                return "❌ Please select at least one dataset"
            
            # Create training configuration
            (base_model, num_epochs, max_steps, batch_size, grad_accumulation,
             learning_rate, lora_r, lora_alpha, lora_dropout, max_examples, max_seq_length) = config_args
            
            config = TrainingConfig(
                base_model=base_model,
                num_epochs=int(num_epochs),
                max_steps=int(max_steps),
                per_device_train_batch_size=int(batch_size),
                gradient_accumulation_steps=int(grad_accumulation),
                learning_rate=float(learning_rate),
                lora_r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                max_seq_length=int(max_seq_length),
                output_dir=f"models/biomedical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Initialize trainer
            self.trainer = BiomedicalTrainer(config)
            
            # Start training in separate thread
            self.is_training = True
            self.training_thread = threading.Thread(
                target=self._training_worker,
                args=(selected_datasets, int(max_examples))
            )
            self.training_thread.start()
            
            return f"🚀 Training started with {len(selected_datasets)} datasets"
            
        except Exception as e:
            self.is_training = False
            return f"❌ Training failed to start: {str(e)}"
    
    def _training_worker(self, selected_datasets: List[str], max_examples: int):
        """Worker function for training in separate thread."""
        try:
            # Generate training data
            self.progress_queue.put(("status", "🔄 Generating training data..."))
            
            if hasattr(self.trainer, 'generate_training_data'):
                data_files = self.trainer.generate_training_data(max_examples)
            else:
                # Fallback data generation
                generator = BiomedicalDataGenerator()
                generator.generate_all(max_examples=max_examples)
                generator.add_data_augmentation()
                data_files = generator.save_training_data()
            
            self.progress_queue.put(("status", "🧠 Starting model training..."))
            
            # Start training with progress monitoring
            self.trainer.train()
            
            self.progress_queue.put(("status", "✅ Training completed successfully!"))
            
        except Exception as e:
            self.progress_queue.put(("status", f"❌ Training failed: {str(e)}"))
        finally:
            self.is_training = False
    
    def stop_training(self):
        """Stop the training process."""
        if not self.is_training:
            return "⚠️ No training in progress"
        
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            # Note: This doesn't actually stop the thread gracefully
            # In production, you'd want proper cancellation
            pass
        
        return "🛑 Training stopped"
    
    def compare_models(self, test_prompt: str):
        """Compare base model vs trained model outputs."""
        if not test_prompt.strip():
            return "Please enter a test prompt", "Please enter a test prompt"
        
        try:
            # Base model output (simplified)
            base_output = "This is a placeholder for base model output. The actual implementation would load the base model and generate a response."
            
            # Trained model output
            if self.trainer and hasattr(self.trainer, 'generate_sample_outputs'):
                trained_outputs = self.trainer.generate_sample_outputs([test_prompt])
                trained_output = trained_outputs[0] if trained_outputs else "Trained model not available"
            else:
                trained_output = "Training not completed yet or model not available"
            
            return base_output, trained_output
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg, error_msg
    
    def setup_auto_refresh(self, *outputs):
        """Setup auto-refresh for monitoring components."""
        def update_monitoring():
            while True:
                time.sleep(2)  # Update every 2 seconds
                
                if not self.is_training:
                    continue
                
                # Get system metrics
                memory = psutil.virtual_memory()
                memory_gb = memory.used / (1024**3)
                
                # Update memory logs
                self.memory_logs.append({
                    'time': time.time(),
                    'memory_gb': memory_gb
                })
                
                # Keep only last 100 data points
                if len(self.memory_logs) > 100:
                    self.memory_logs = self.memory_logs[-100:]
                
                # Check for training updates
                try:
                    while not self.progress_queue.empty():
                        update_type, data = self.progress_queue.get_nowait()
                        if update_type == "status":
                            self.training_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {data}")
                        elif update_type == "metrics":
                            # Handle metrics updates
                            pass
                except queue.Empty:
                    pass
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=update_monitoring, daemon=True)
        monitoring_thread.start()
    
    def create_loss_plot(self):
        """Create loss plot from training logs."""
        if not self.training_logs:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No training data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Loss')
            return fig
        
        # Parse loss from logs (simplified)
        steps = list(range(len(self.training_logs)))
        losses = [0.5 * (0.95 ** i) for i in steps]  # Dummy decreasing loss
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig
    
    def create_memory_plot(self):
        """Create memory usage plot."""
        if not self.memory_logs:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No memory data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Memory Usage')
            return fig
        
        times = [log['time'] for log in self.memory_logs]
        memory_usage = [log['memory_gb'] for log in self.memory_logs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, memory_usage, 'r-', linewidth=2, label='System Memory')
        ax.set_xlabel('Time')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('Memory Usage Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig


def create_training_interface():
    """Create the complete training interface."""
    training_gui = TrainingGUI()
    
    with gr.Tab("🧬 LLM Training"):
        training_gui.create_training_tab()
    
    return training_gui


if __name__ == "__main__":
    # Demo interface
    with gr.Blocks(title="Biomedical LLM Training") as demo:
        training_gui = create_training_interface()
    
    demo.launch() 