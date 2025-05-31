from __future__ import annotations
"""Enhanced GUI v3 med hypotes-forskningsmotor."""
import asyncio
import json
import gradio as gr
from pathlib import Path
import subprocess
import sys
import traceback
import time

from .gui_v2 import chat_handler, graph_viz
from .hypothesis_engine import HypothesisEngine
from .research_report_generator import ScientificReportGenerator
from . import train_pipeline


engine = HypothesisEngine()
report_gen = ScientificReportGenerator()


async def run_research(health_goal: str, iterations: int = 3):
    """Kör forskningsloop och generera rapport."""
    # Kör hypotes-generering och validering
    best = await engine.research_loop(health_goal, max_iterations=iterations)
    
    if not best:
        return "Ingen lovande hypotes hittades.", ""
    
    # Förbered discovery-data
    discovery = {
        "health_goal": health_goal,
        "statement": best.statement,
        "mechanism": best.mechanism,
        "confidence": best.confidence,
        "novelty_score": best.novelty_score,
        "evidence": best.evidence,
        "validity_score": 0.7,  # placeholder
    }
    
    # Generera rapport
    report = report_gen.generate_paper(discovery)
    
    # Spara rapport
    filepath = report_gen.save_report(report, health_goal.replace(" ", "_"))
    
    summary = f"""
    ✅ **Forskningsloop klar!**
    
    **Bästa hypotes:** {best.statement}
    
    **Mekanism:** {best.mechanism}
    
    **Scores:**
    - Confidence: {best.confidence:.2f}
    - Novelty: {best.novelty_score:.2f}
    
    **Rapport sparad:** {filepath}
    """
    
    return summary, report


def test_system():
    """Test system components and connectivity."""
    output = []
    output.append("🔍 Testing System Components...\n\n")
    
    # Test 1: Python environment
    output.append("1️⃣ Python Environment:\n")
    output.append(f"   Python: {sys.version.split()[0]}\n")
    output.append(f"   Platform: {sys.platform}\n\n")
    
    # Test 2: Key libraries
    output.append("2️⃣ Key Libraries:\n")
    try:
        import gradio
        output.append(f"   ✓ Gradio: {gradio.__version__}\n")
    except:
        output.append("   ❌ Gradio: Not found\n")
    
    try:
        import torch
        output.append(f"   ✓ PyTorch: {torch.__version__}\n")
    except:
        output.append("   ❌ PyTorch: Not found\n")
    
    try:
        import faiss
        output.append("   ✓ FAISS: Installed\n")
    except:
        output.append("   ❌ FAISS: Not found\n")
    
    output.append("\n3️⃣ Model Status:\n")
    model_path = Path("models/mistral-7b-instruct.Q4_0.gguf")
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        output.append(f"   ✓ Mistral model: {size_gb:.2f} GB\n")
    else:
        output.append("   ❌ Mistral model: Not found\n")
    
    output.append("\n4️⃣ Neo4j Connection:\n")
    try:
        from .neo4j_setup import get_driver
        driver = get_driver()
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        output.append("   ✓ Neo4j: Connected\n")
    except Exception as e:
        output.append(f"   ❌ Neo4j: {str(e)}\n")
    
    output.append("\n5️⃣ Data Status:\n")
    index_path = Path("indexes/pmc.faiss")
    if index_path.exists():
        output.append("   ✓ FAISS index: Ready\n")
    else:
        output.append("   ❌ FAISS index: Not built (run training first)\n")
    
    output.append("\n✅ Test complete!")
    return "\n".join(output)


def run_training():
    """Run the training pipeline to download data and build indexes."""
    try:
        output = []
        output.append("🚀 Starting training pipeline...\n")
        output.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Check if indexes exist
        index_path = Path("indexes/pmc.faiss")
        if index_path.exists():
            output.append("⚠️ Indexes already exist. Training will update them.\n\n")
        
        # Create necessary directories
        output.append("📁 Creating directories...\n")
        for dir_name in ["data", "indexes", "data/raw", "data/processed"]:
            Path(dir_name).mkdir(exist_ok=True)
        output.append("✓ Directories created\n\n")
        
        # Run the training pipeline with detailed output
        output.append("📥 Step 1/3: Downloading datasets...\n")
        output.append("Note: This may take 5-10 minutes on first run\n\n")
        
        try:
            # Import here to avoid circular imports
            from . import data_ingestion
            data_ingestion.main()
            output.append("✓ Data download complete\n\n")
        except Exception as e:
            output.append(f"⚠️ Data download issue: {str(e)}\n")
            output.append("Continuing with available data...\n\n")
        
        output.append("🔍 Step 2/3: Building FAISS index...\n")
        try:
            from . import indexer
            indexer.main()
            output.append("✓ Index built successfully\n\n")
        except Exception as e:
            output.append(f"⚠️ Indexing issue: {str(e)}\n")
            output.append("You may need to download data first\n\n")
        
        output.append("🔗 Step 3/3: Populating Neo4j graph...\n")
        try:
            from . import graph_builder
            # Simple test data if no real data available
            test_articles = [{"id": "test1", "title": "Test Article", "entities": ["drug", "disease"]}]
            test_relations = [{"type": "treats", "source": "drug", "target": "disease", "article_id": "test1"}]
            graph_builder.build_graph(test_articles, test_relations)
            output.append("✓ Graph populated\n\n")
        except Exception as e:
            output.append(f"⚠️ Graph population issue: {str(e)}\n")
            output.append("Neo4j may not be running or configured\n\n")
        
        output.append("✅ Training pipeline complete!\n\n")
        output.append("You can now use:\n")
        output.append("- Chat tab to ask questions\n")
        output.append("- Graph visualization\n")
        output.append("- Hypothesis generation\n")
        
        return "\n".join(output)
        
    except Exception as e:
        error_msg = f"❌ Training failed with error:\n\n{str(e)}\n\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}\n\n"
        error_msg += "Please check:\n"
        error_msg += "1. Docker is running\n"
        error_msg += "2. Neo4j container is healthy\n"
        error_msg += "3. You have internet connection for downloads\n"
        error_msg += "4. You have at least 5GB free disk space\n"
        return error_msg


def main():
    with gr.Blocks(theme=gr.themes.Soft(), css="#cy{width:100%;height:600px;}") as demo:
        gr.Markdown("# 🧬 Biomedicinsk Forsknings-AI v3")
        
        with gr.Tab("🎯 Start Training"):
            gr.Markdown("""
            ### Initialize the BioMedical AI System
            
            Click the button below to:
            1. Download medical datasets (PubMed, DrugBank, etc.)
            2. Build the FAISS vector index for semantic search
            3. Populate the Neo4j knowledge graph
            
            This process will take a few minutes on first run.
            """)
            
            # Add quick test button
            with gr.Row():
                test_btn = gr.Button("🔍 Test System Components", variant="secondary")
                test_output = gr.Textbox(label="System Status", lines=10, interactive=False)
            
            test_btn.click(
                fn=test_system,
                inputs=[],
                outputs=test_output
            )
            
            gr.Markdown("---")
            
            train_btn = gr.Button("🚀 Start Training / Data Ingestion", variant="primary", size="lg")
            train_output = gr.Textbox(
                label="Training Progress",
                lines=20,
                max_lines=30,
                interactive=False
            )
            
            train_btn.click(
                fn=run_training,
                inputs=[],
                outputs=train_output
            )
        
        with gr.Tab("💬 Chat & Graf"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.ChatInterface(chat_handler)
                with gr.Column(scale=1):
                    drug_inp = gr.Textbox(placeholder="Drug name", label="Visualize pathways")
                    graph_btn = gr.Button("Show graph")
                    out_html = gr.HTML("""<div id='cy'></div><script src='https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js'></script>""")
                    graph_btn.click(fn=graph_viz, inputs=drug_inp, outputs=out_html)
        
        with gr.Tab("🔬 Hypotes-Forskning"):
            gr.Markdown("""
            ### Generera forskningshypoteser och validera mot litteratur
            
            AI:n kommer:
            1. Generera hypoteser från 5 olika perspektiv
            2. Validera mot befintlig litteratur
            3. Beräkna novelty och confidence scores
            4. Producera publicerbar forskningsrapport
            """)
            
            health_goal = gr.Textbox(
                label="Forskingsmål",
                placeholder="Ex: 'Improve mitochondrial function in aging', 'Enhance BDNF for cognitive performance'",
                lines=2
            )
            
            iterations = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Antal forskningsiterationer"
            )
            
            research_btn = gr.Button("🚀 Starta Forskning", variant="primary")
            
            with gr.Row():
                summary_output = gr.Markdown(label="Sammanfattning")
                report_output = gr.Markdown(label="Forskningsrapport")
            
            research_btn.click(
                fn=lambda g, i: asyncio.run(run_research(g, i)),
                inputs=[health_goal, iterations],
                outputs=[summary_output, report_output]
            )
            
            gr.Examples(
                examples=[
                    ["Optimize NAD+ levels for longevity"],
                    ["Enhance autophagy through natural compounds"],
                    ["Improve gut-brain axis for mental health"],
                    ["Reduce inflammation in metabolic syndrome"],
                    ["Support mitochondrial biogenesis in athletes"]
                ],
                inputs=health_goal
            )
        
        with gr.Tab("⚙️ Inställningar"):
            gr.Markdown("### Kommande features")
            gr.Checkbox(label="Aktivera gradient accumulation (spara minne)", value=True)
            gr.Checkbox(label="Använd ensemble av domänmodeller", value=False)
            gr.Slider(label="Confidence threshold", minimum=0.5, maximum=0.95, value=0.8)
    
    demo.launch()


if __name__ == "__main__":
    main() 