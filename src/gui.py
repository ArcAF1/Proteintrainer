"""Gradio chat interface.

Usage example:
    python src/gui.py
"""
import asyncio
import subprocess
from pathlib import Path

import gradio as gr

from .rag_chat import answer
from .config import settings


def chat(query: str, history: list[tuple[str, str]]) -> str:
    return asyncio.run(answer(query))


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("# Offline Medical Assistant\n*Not medical advice*")
        gr.ChatInterface(fn=chat)
        file_in = gr.File(label="Training JSONL")
        train_btn = gr.Button("Train Model")
        status = gr.Markdown()

        def train(file):
            if file is None:
                return "No file selected"
            subprocess.Popen([
                "python",
                "src/fine_tune.py",
                "--base-model",
                str(Path("models") / settings.llm_model),
                "--train-file",
                file.name,
            ])
            return "Training started..."

        train_btn.click(train, inputs=file_in, outputs=status)
    demo.launch()


if __name__ == "__main__":
    main()

