"""Gradio chat interface.

Usage example:
    python src/gui.py
"""
import asyncio
import gradio as gr

from .rag_chat import answer


def chat(query: str, history: list[tuple[str, str]]) -> str:
    return asyncio.run(answer(query))


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("# Offline Medical Assistant\n*Not medical advice*")
        gr.ChatInterface(fn=chat)



    demo.launch()




if __name__ == "__main__":
    main()

