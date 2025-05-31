from __future__ import annotations
import asyncio
import json
import gradio as gr
from gradio.components import HTML
from .rag_chat import answer as rag_answer
from .graph_intelligence import BiomedicalGraphQuerier
from .train_pipeline import main as run_train
from pathlib import Path

qy = BiomedicalGraphQuerier()


def chat_handler(query: str, history):
    try:
        # Check if indexes exist
        index_path = Path("indexes/pmc.faiss")
        if not index_path.exists():
            return "⚠️ The system needs to be initialized first. Please go to the 'Start Training' tab and click the training button to download data and build indexes."
        
        resp = asyncio.run(rag_answer(query))
        # simple parse citations: assume "Sources:" block json placeholder
        return resp
    except FileNotFoundError as e:
        return f"⚠️ System not initialized: {str(e)}\n\nPlease go to the 'Start Training' tab first."
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check the system status and try again."


def graph_viz(drug_name: str):
    try:
        if not drug_name:
            return "Please enter a drug name"
            
        res = qy.find_drug_pathways(drug_name)
        nodes = {}
        edges = []
        for path in res:
            prev = None
            for n in path["nodes"]:
                nodes[n] = {"data": {"id": n, "label": n}}
                if prev:
                    edge_id = f"{prev}-{n}"
                    edges.append({"data": {"id": edge_id, "source": prev, "target": n}})
                prev = n
        elements = list(nodes.values()) + edges
        
        if not elements:
            return f"No pathways found for '{drug_name}'. Try another drug name."
            
        js = f"window.cy && cy.destroy();cy = cytoscape({{container: document.getElementById('cy'), elements: {json.dumps(elements)}}});"
        return js
    except Exception as e:
        return f"Error visualizing graph: {str(e)}"


def main():
    with gr.Blocks(css="#cy{width:100%;height:600px;}") as demo:
        gr.Markdown("# 🔬 Biomedical Assistant v2")
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.ChatInterface(chat_handler)
            with gr.Column(scale=1):
                drug_inp = gr.Textbox(placeholder="Drug name", label="Visualize pathways")
                graph_btn = gr.Button("Show graph")
                out_html = HTML("""<div id='cy'></div><script src='https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js'></script><script>var cy;</script>""")
                graph_btn.click(fn=graph_viz, inputs=drug_inp, outputs=out_html, js=True)
    demo.launch()

if __name__ == "__main__":
    main() 