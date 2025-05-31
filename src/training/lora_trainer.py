"""CLI helper for LoRA fine-tuning of Mistral-7B (QLoRA 4-bit).

Usage example:
    python -m src.training.lora_trainer \
      --base-model mistralai/Mistral-7B-Instruct-v0.2 \
      --data-jsonl data/phase2_corpus.jsonl \
      --output models/biomedical_mistral_lora

The script is **optimised for Apple-Silicon (MPS)** but falls back to CPU.
It is *not* used by the GUI directly – the GUI calls the higher-level
BiomedicalTrainer which already wraps similar logic – but having this script
lets power-users run a standalone fine-tune.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch  # type: ignore
from datasets import load_dataset, Dataset  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer  # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from peft import PeftModel
from transformers import BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl_as_dataset(path: str | Path) -> Dataset:
    """Very small helper that reads an instruction-tuning JSONL file."""
    import json
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main():  # noqa: D401 – CLI entry point
    parser = argparse.ArgumentParser(description="LoRA fine-tune helper (QLoRA 4-bit)")
    parser.add_argument("--base-model", required=True, help="Base HF model e.g. mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--data-jsonl", required=True, help="JSONL with instruction-tuning data (Alpaca format)")
    parser.add_argument("--output", required=True, help="Directory for LoRA adapters")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[lora_trainer] Device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16 if device == "cpu" else torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_cfg)

    train_dataset = _load_jsonl_as_dataset(args.data_jsonl)

    def _tokenise(batch):
        return tokenizer(
            batch["instruction"] + "\n" + batch.get("input", ""),
            truncation=True,
            padding="max_length",
            max_length=2048,
        )

    train_dataset = train_dataset.map(_tokenise, batched=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        fp16=device != "cpu",
        bf16=device == "cpu",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()

    # save adapters only
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("[lora_trainer] ✅ training complete; adapters saved →", args.output)


if __name__ == "__main__":
    main() 