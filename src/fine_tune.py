"""LoRA fine-tuning script.

Usage example:
    python src/fine_tune.py --train-file data/train.jsonl --base-model models/mistral-7b-instruct.Q4_0.gguf
"""
from __future__ import annotations

import argparse
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

from .config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=settings.model_dir / "lora" )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=str(args.train_file))
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    model = AutoModelForCausalLM.from_pretrained(str(args.base_model))

    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(model, config)

    def tokenize(batch):
        return tokenizer(batch["prompt"], truncation=True)

    tokenized = dataset.map(tokenize)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"], data_collator=data_collator)
    trainer.train()
    model.save_pretrained(str(args.output))


if __name__ == "__main__":
    main()
