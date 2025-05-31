"""Gradio chat interface.

Usage example:
    python src/gui.py
"""
import asyncio
import subprocess
import sys

import gradio as gr

from .rag_chat import answer
from . import graph_query
from .train_pipeline import main as run_train_pipeline
from .graph_rag import GraphRAG
from .memory_manager import save_finding, recall

# instantiate graphRAG if available
try:
    _graph_rag = GraphRAG() if GraphRAG.available else None
except Exception:  # pragma: no cover
    _graph_rag = None


def chat(query: str, history: list[tuple[str, str]]):
    """Route chat or slash commands."""
    if query.startswith("/graph "):
        name = query.split(maxsplit=1)[1]
        lines = graph_query.search_entity(name)
        return "\n".join(lines) or f"Inga träffar för {name}."
    if query.startswith("/save"):
        # save last answer in history
        if history:
            q, a = history[-1]
            save_finding(q, a)
            return "Saved to memory."
        return "Nothing to save yet."

    # Normal chat flow
    previous_notes = recall(query)
    notes_context = "\n".join(previous_notes)
    if notes_context:
        query_for_model = f"User notes:\n{notes_context}\n\nQuestion: {query}"
    else:
        query_for_model = query

    if _graph_rag is not None:
        answer_text = _graph_rag.answer(query_for_model)
    else:
        answer_text = asyncio.run(answer(query_for_model))

    # store finding automatically
    save_finding(query, answer_text)
    return answer_text


def train_button_click():
    """Long-running training pipeline launched from UI."""
    try:
        run_train_pipeline()
        return "✅ Training finished."
    except Exception as exc:  # pylint: disable=broad-except
        return f"❌ Training failed: {exc}"


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("# Offline Medical Assistant\n*Not medical advice*")
        status = gr.Textbox(label="Status", interactive=False)
        train_btn = gr.Button("Train / Learn")
        train_btn.click(fn=train_button_click, outputs=status)
        gr.ChatInterface(fn=chat)

        demo.queue(concurrency_count=2)
    demo.launch()


if __name__ == "__main__":
    main()

