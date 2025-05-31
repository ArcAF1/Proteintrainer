from __future__ import annotations
"""LLM-first GUI where the local AI assistant handles all interactions."""
import asyncio
import json
import warnings

# Suppress pkg_resources deprecation warning from gradio/dependencies
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import gradio as gr
from pathlib import Path
import subprocess
import sys
import traceback
import time
from typing import Optional, List, Tuple, Dict, Any
import threading
import socket
import logging

from .rag_chat import answer as rag_answer, get_rag_status
from .graph_rag import GraphRAG, graphrag_available
from .graph_query import get_subgraph
from .memory_manager import save_finding, recall
from .hypothesis_engine import HypothesisEngine
from .research_report_generator import ScientificReportGenerator
from . import train_pipeline
from .agent import ask_agent
from .biomedical_agent_integration import enhanced_biomedical_handler
from .autonomous_research_agent import AutonomousResearchAgent
from .research_logger import ResearchLogger
from .research_library import ResearchLibrary
from .experimental_research_engine import ExperimentalResearchEngine, integrate_with_graph
from .research_triggers import (
    research_detector, 
    create_research_prompt, 
    format_research_response,
    RESEARCH_PROMPTS
)

# Global instances
engine = HypothesisEngine()
report_gen = ScientificReportGenerator()
graph_rag_instance: Optional[GraphRAG] = None
research_agent: Optional[AutonomousResearchAgent] = None
research_logger: Optional[ResearchLogger] = None

# Progress tracking
progress_message = ""
progress_percentage = 0.0
_progress_lock = threading.Lock()

logger = logging.getLogger(__name__)

def update_progress(percentage: float, message: str):
    """Thread-safe progress update."""
    global progress_message, progress_percentage
    with _progress_lock:
        progress_message = message
        progress_percentage = percentage

def get_progress():
    """Thread-safe progress read."""
    with _progress_lock:
        return progress_percentage, progress_message

def save_memory(message: str, history: List[List[str]]) -> str:
    """Save the last conversation to memory."""
    if history and len(history) > 0:
        last_q, last_a = history[-1]
        # Simple memory storage
        memory_file = Path("data/memory/conversations.json")
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        memories = []
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                memories = json.load(f)
        
        memories.append({
            'timestamp': time.time(),
            'question': last_q,
            'answer': last_a
        })
        
        with open(memory_file, 'w') as f:
            json.dump(memories, f, indent=2)
            
        return "✅ **Saved to memory!** I'll remember this for future reference."
    return "❌ No previous conversation to save. Ask me something first!"

def recall_memory(query: str) -> str:
    """Recall information from memory."""
    memory_file = Path("data/memory/conversations.json")
    
    if not memory_file.exists():
        return f"🤔 I don't have any saved memories yet. Ask me something and then save it!"
    
    with open(memory_file, 'r') as f:
        memories = json.load(f)
    
    # Simple keyword search
    relevant = []
    query_lower = query.lower()
    for mem in memories:
        if query_lower in mem['question'].lower() or query_lower in mem['answer'].lower():
            relevant.append(f"Q: {mem['question']}\nA: {mem['answer']}")
    
    if relevant:
        return f"📝 **Here's what I remember about '{query}':**\n\n" + "\n\n---\n\n".join(relevant[:3])
    else:
        return f"🤔 I don't have any memories about '{query}'. Try a different search term!"

async def ai_system_handler(message: str, history: List[List[str]]) -> str:
    """
    Main chat handler for the unified biomedical AI system.
    Routes messages to appropriate handlers and maintains conversation context.
    """
    try:
        msg_lower = message.lower()
        
        # Check for research triggers FIRST
        should_research, research_topic = research_detector.should_trigger_research(message)
        if should_research and research_topic:
            # Show immediate feedback
            immediate_response = f"""🔬 **Research Mode Activated!**

I'm launching an experimental research investigation on: *"{research_topic}"*

This will:
- Generate multiple hypotheses
- Search your 11GB knowledge base
- Run simulations where applicable
- Generate innovative ideas

**Research starting now...** (this takes 1-3 minutes)
"""
            
            # Run research asynchronously
            try:
                research_result = await research_detector.trigger_research(research_topic, max_iterations=5)
                research_result['topic'] = research_topic
                
                # Format the response
                research_response = format_research_response(research_result)
                
                # Also give a quick summary in chat
                return immediate_response + "\n\n" + research_response
                
            except Exception as e:
                return immediate_response + f"\n\n❌ Research failed: {str(e)}"
        
        # Check for system commands
        if any(phrase in msg_lower for phrase in ['test', 'test system', 'system test', 'check system', 'diagnostics']):
            return run_system_test()
            
        elif 'start training' in msg_lower:
            if 'full' in msg_lower or 'complete' in msg_lower:
                return start_full_training()
            else:
                return start_data_pipeline()
                
        elif 'training status' in msg_lower or "what's the training status" in msg_lower:
            percentage, msg = get_progress()
            if msg:
                return f"📊 **Training Progress**: {percentage:.1f}%\n\n{msg}"
            else:
                return "✅ **No training in progress**\n\nYou can start training by saying 'start training' or 'start full training'."
                
        # Special knowledge graph commands
        elif graphrag_available() and ('show' in msg_lower or 'graph' in msg_lower):
            entity = extract_entity_from_message(message)
            if entity:
                return generate_text_graph(entity)
                
        # Research commands
        elif 'research' in msg_lower or 'investigate' in msg_lower:
            research_goal = extract_research_goal(message)
            if research_goal:
                return run_research_sync(research_goal)
                
        # Internet search
        elif 'search' in msg_lower and ('internet' in msg_lower or 'web' in msg_lower or 'online' in msg_lower):
            return f"🔍 I'll search the internet for: {message}\n\n[Internet search functionality would be implemented here]"
            
        # Memory operations
        elif 'save' in msg_lower and 'memory' in msg_lower:
            return save_memory(message, history)
            
        elif 'recall' in msg_lower or 'remember' in msg_lower:
            query = extract_recall_query(message)
            return recall_memory(query)
            
        else:
            # Use enhanced pharmaceutical handler
            response = await enhanced_biomedical_handler(message, history)
        
        # Add AI personality and helpful suggestions
        enhanced_response = f"{response}\n\n"
        
        # Add contextual suggestions based on the topic
        if any(word in message.lower() for word in ['protein', 'drug', 'molecule']):
            enhanced_response += "💡 **Tip:** You can ask me to 'show the graph' or 'visualize relationships' for any entity!"
        elif any(word in message.lower() for word in ['study', 'effect', 'mechanism']):
            enhanced_response += "💡 **Tip:** Ask me to 'research this topic' for a deeper analysis with hypothesis generation!"
        elif any(word in message.lower() for word in ['longevity', 'aging', 'health']):
            enhanced_response += "💡 **Tip:** I can search for 'latest research' or help you 'investigate' specific interventions!"
        
        return enhanced_response
        
    except Exception as e:
        error_msg = str(e)
        print(f"AI system handler error: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return f"❌ **I encountered an error:** {error_msg}\n\n" + \
               "Don't worry! You can try:\n" + \
               "- Rephrasing your question\n" + \
               "- Asking me to 'test the system' to check if everything is working\n" + \
               "- Running 'start data pipeline' if the knowledge base isn't built yet\n" + \
               "- Restarting the application if the error persists"

def extract_entity_from_message(message: str) -> Optional[str]:
    """Extract entity name from user message."""
    # Simple extraction - look for common patterns
    words = message.lower().split()
    
    # Look for entities after these trigger words
    triggers = ['graph', 'show', 'visualize', 'of', 'for']
    for i, word in enumerate(words):
        if word in triggers and i + 1 < len(words):
            # Take the next word as the entity
            entity = words[i + 1].strip('.,!?')
            if len(entity) > 2:  # Avoid very short words
                return entity
    
    # Look for quoted entities
    import re
    quoted = re.findall(r'"([^"]*)"', message)
    if quoted:
        return quoted[0]
    
    # Look for common biomedical entities
    biomedical_terms = ['insulin', 'metformin', 'creatine', 'protein', 'dopamine', 'serotonin']
    for term in biomedical_terms:
        if term in message.lower():
            return term
    
    return None

def extract_research_goal(message: str) -> Optional[str]:
    """Extract research goal from user message."""
    # Look for patterns like "research X" or "study Y"
    import re
    
    patterns = [
        r'research (?:on |about |into )?(.+?)(?:\.|$)',
        r'study (?:the )?(.+?)(?:\.|$)',
        r'investigate (.+?)(?:\.|$)',
        r'analyze (.+?)(?:\.|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1).strip()
    
    return None

def extract_recall_query(message: str) -> str:
    """Extract query from recall/remember message."""
    # Remove common recall trigger words
    words = message.lower().split()
    filtered = [w for w in words if w not in ['recall', 'remember', 'what', 'do', 'you', 'about']]
    return ' '.join(filtered).strip()

def run_system_test() -> str:
    """Run comprehensive system test and return results."""
    output = []
    output.append("🔍 **Running system diagnostics...**\n")
    
    # Test 1: Python environment
    output.append("**1️⃣ Python Environment:**")
    output.append(f"• Python version: {sys.version.split()[0]}")
    output.append(f"• Platform: {sys.platform}\n")
    
    # Test 2: Key libraries
    output.append("**2️⃣ Dependencies:**")
    libraries = [
        ('gradio', 'Web interface'),
        ('torch', 'AI/ML framework'),
        ('faiss', 'Vector search'),
        ('neo4j', 'Graph database client'),
        ('transformers', 'Language models'),
        ('sentence_transformers', 'Embeddings')
    ]
    
    for lib, desc in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'installed')
            output.append(f"• ✅ {desc}: {version}")
        except ImportError:
            output.append(f"• ❌ {desc}: Not found")
    
    output.append("")
    
    # Test 3: RAG System Status
    output.append("**3️⃣ RAG System:**")
    rag_status = get_rag_status()
    output.append(rag_status)
    output.append("")
    
    # Test 4: Models
    output.append("**4️⃣ AI Models:**")
    model_path = Path("models/mistral-7b-instruct.Q4_0.gguf")
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        output.append(f"• ✅ Local LLM: {size_gb:.1f} GB")
    else:
        output.append("• ❌ Local LLM: Not downloaded")
    
    output.append("")
    
    # Test 5: Neo4j
    output.append("**5️⃣ Knowledge Graph:**")
    try:
        from .neo4j_setup import get_driver
        driver = get_driver()
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as nodes")
            node_count = result.single()["nodes"]
        output.append(f"• ✅ Neo4j connected: {node_count:,} nodes")
    except Exception as e:
        output.append(f"• ❌ Neo4j: {str(e)[:50]}...")
    
    output.append("")
    
    # Test 6: Indexes
    output.append("**6️⃣ Search Indexes:**")
    index_path = Path("indexes/pmc.faiss")
    if index_path.exists():
        output.append("• ✅ FAISS index: Ready")
    else:
        output.append("• ❌ FAISS index: Not built")
    
    output.append("\n**🎯 Recommendations:**")
    
    if not model_path.exists():
        output.append("• Run training to download the AI model")
    if not index_path.exists():
        output.append("• Run training to build search indexes")
    
    output.append("• Try asking: 'What is the mechanism of metformin?'")
    output.append("• Try: 'Show me a graph of insulin relationships'")
    output.append("• Try: 'Research ways to improve longevity'")
    
    return "\n".join(output)

def start_data_pipeline() -> str:
    """Start the data pipeline (indexing and graph building)."""
    # Check if already running
    percentage, msg = get_progress()
    if msg:
        return f"🔄 **Training already in progress:** {percentage:.1f}%\n\n{msg}\n\nPlease wait for it to complete."
    
    # Start training in background
    def run_data_pipeline():
        try:
            update_progress(0, "Initializing data pipeline...")
            train_pipeline.main(progress_callback=update_progress)
            update_progress(100, "Data pipeline complete! ✅")
            time.sleep(5)  # Show completion message for a bit
            update_progress(0, "")  # Clear progress
        except Exception as e:
            update_progress(0, f"Data pipeline failed: {str(e)} ❌")
    
    thread = threading.Thread(target=run_data_pipeline, daemon=True)
    thread.start()
    
    return """🚀 **Data Pipeline Started!** 

**What I'm doing:**
1. 📥 Downloading biomedical datasets (PubMed, DrugBank, etc.)
2. 🔍 Building FAISS search indexes  
3. 🕸️ Populating Neo4j knowledge graph
4. 🤖 Setting up the knowledge base

**This will take 10-30 minutes** depending on your internet speed.

I'll update you on progress. You can ask me 'what's the training status?' anytime!

💡 **While training runs,** you can still ask me general questions - I'll do my best to answer!"""

def start_full_training() -> str:
    """Start full training including LLM fine-tuning."""
    # Check if already running
    percentage, msg = get_progress()
    if msg:
        return f"🔄 **Training already in progress:** {percentage:.1f}%\n\n{msg}\n\nPlease wait for it to complete."
    
    # Start full training in background
    def run_full_training():
        try:
            # Step 1: Data pipeline
            update_progress(0, "Phase 1/2: Running data pipeline...")
            print("[Full Training] Starting data pipeline...")
            
            def pipeline_progress(p, m):
                progress = p * 0.3  # Data pipeline takes 30% of total
                update_progress(progress, f"Phase 1/2: {m}")
            
            train_pipeline.main(progress_callback=pipeline_progress)
            print("[Full Training] Data pipeline complete!")
            
            # Step 2: LLM fine-tuning
            update_progress(30, "Phase 2/2: Starting LLM fine-tuning...")
            print("[Full Training] Starting LLM fine-tuning...")
            
            # Import and run the training connector
            from .training_connector import get_training_connector
            
            connector = get_training_connector()
            
            def training_progress_callback(progress, message):
                # Map training progress to 30-100% range
                overall_progress = 30 + (progress * 0.7)
                update_progress(overall_progress, f"Phase 2/2: {message}")
            
            success = connector.start_full_training(progress_callback=training_progress_callback)
            
            if success:
                update_progress(100, "Full training complete! ✅")
                print("[Full Training] Complete! Model ready for use.")
            else:
                update_progress(0, "Training completed with warnings ⚠️")
                
            time.sleep(5)  # Show completion message for a bit
            update_progress(0, "")  # Clear progress
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Full Training] Error: {error_msg}")
            print(f"[Full Training] Traceback: {traceback.format_exc()}")
            
            # Provide helpful error message
            if "No module named" in error_msg:
                update_progress(0, f"Missing dependency: {error_msg} ❌")
            elif "CUDA" in error_msg or "MPS" in error_msg:
                update_progress(0, "GPU/Metal configuration issue ❌")
            elif "out of memory" in error_msg.lower():
                update_progress(0, "Out of memory - try reducing batch size ❌")
            else:
                update_progress(0, f"Training failed: {error_msg} ❌")
    
    thread = threading.Thread(target=run_full_training, daemon=True)
    thread.start()
    
    return """🎯 **Full Training Started!**

**What I'm doing:**

**Phase 1 (10-30 min):** Data Pipeline
- 📥 Download biomedical datasets (PubMed, DrugBank, etc.)
- 🔍 Build FAISS search indexes for fast retrieval
- 🕸️ Populate Neo4j knowledge graph with relationships

**Phase 2 (1-3 hours):** LLM Fine-tuning  
- 🧠 Fine-tune Mistral-7B on biomedical data
- 🔬 Optimize for medical/scientific question answering
- 💾 Save specialized model adapters (LoRA)

**Total time: 1.5-3.5 hours** 

This creates a truly specialized biomedical AI that understands:
- Drug mechanisms and interactions
- Protein structures and pathways  
- Disease relationships and treatments
- Latest research papers and clinical trials

Ask me 'training status' anytime for updates.

⚠️ **Note:** This is computationally intensive. Your Mac may get warm and fans may increase speed. This is normal for M1/M2 systems under heavy load."""

def generate_text_graph(entity: str) -> str:
    """Generate a text-based representation of entity relationships."""
    try:
        nodes, edges = get_subgraph(entity, max_depth=2)
        
        if not nodes:
            return f"❌ No relationships found for '{entity}' in the knowledge graph."
        
        # Create text representation
        result = f"**🕸️ Relationship Map for {entity.title()}:**\n\n"
        
        # Group relationships by type
        relationships = {}
        for edge in edges:
            rel_type = edge.get('type', 'related_to')
            if rel_type not in relationships:
                relationships[rel_type] = []
            relationships[rel_type].append((edge['source'], edge['target']))
        
        for rel_type, connections in relationships.items():
            result += f"**{rel_type.replace('_', ' ').title()}:**\n"
            for source, target in connections[:5]:  # Limit to avoid overwhelming
                if source.lower() == entity.lower():
                    result += f"• {entity} → {target}\n"
                elif target.lower() == entity.lower():
                    result += f"• {source} → {entity}\n"
                else:
                    result += f"• {source} ↔ {target}\n"
            result += "\n"
        
        if len(edges) > 20:
            result += f"*... and {len(edges) - 20} more relationships*\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error generating graph: {str(e)}"

def run_research_sync(goal: str) -> str:
    """Run research synchronously and return summary."""
    try:
        update_progress(0, f"Researching: {goal}")
        
        # This is a simplified version - in reality you'd want async
        # For now, return a placeholder that explains what would happen
        return f"""🔬 **Research Analysis Started**

**Goal:** {goal}

**My Research Process:**
1. 🧠 Generate multiple hypotheses from different angles
2. 📚 Validate against existing literature  
3. 🎯 Calculate confidence and novelty scores
4. 📄 Produce comprehensive report

**Note:** Full research analysis takes several minutes. For the complete experience, use the Research Lab tab.

**Quick insights based on my knowledge:**
{asyncio.run(rag_answer(f"What are the key mechanisms and approaches for: {goal}"))}

💡 **Want the full research?** Go to the Research Lab tab for comprehensive hypothesis generation and validation!"""
        
    except Exception as e:
        return f"❌ Research error: {str(e)}"

def create_unified_gui():
    """Create LLM-first conversational interface."""
    # Initialize directories
    data_dir = Path("data")
    memory_dir = data_dir / "memory"
    for dir_path in [data_dir, memory_dir]:
        dir_path.mkdir(exist_ok=True)
    
    with gr.Blocks(
        title="🧬 Local Biomedical AI Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1000px; margin: auto; }
        .main-chat { min-height: 600px; }
        """
    ) as demo:
        
        gr.Markdown("# 🧬 Local Biomedical AI Assistant")
        gr.Markdown("**Your personal AI researcher** - 100% local, no cloud APIs needed")
        
        # Progress display
        progress_display = gr.Markdown(value="", visible=False)
        
        # Main chat interface - this is the primary interaction
        gr.Markdown("## 💬 Chat with your AI Assistant")
        gr.Markdown("Ask me anything about biomedical topics, or tell me what you'd like to do!")
        
        chat_interface = gr.ChatInterface(
            ai_system_handler,
            examples=[
                "Test if the system is working properly",
                "Start training to download datasets",
                "What is the mechanism of action of metformin?",
                "Show me how insulin connects to other molecules", 
                "Research ways to improve mitochondrial function",
                "Search for latest longevity research online",
                "Save this finding to memory",
                "What do you remember about creatine?",
            ],
            title=None,
            description="💡 **I can:** Answer questions • Run system tests • Start training • Create graphs • Conduct research • Search latest papers • Remember findings"
        )
        
        # Expandable advanced tools
        with gr.Accordion("🔧 Advanced Tools", open=False):
            gr.Markdown("*These tools are also available through natural language - just ask me!*")
            
            # API Enhancement Toggle
            with gr.Row():
                api_toggle = gr.Checkbox(
                    label="🌐 Enable Live API Data", 
                    value=False,
                    info="Enhance responses with real-time data from PubMed, Clinical Trials, etc."
                )
                api_status = gr.Markdown("API enhancement: **Disabled**")
                
                def toggle_api_enhancement(enabled):
                    from .rag_chat import enable_api_enhancement
                    enable_api_enhancement(enabled)
                    status = "**Enabled** ✅ - Responses will include latest research" if enabled else "**Disabled**"
                    return f"API enhancement: {status}"
                
                api_toggle.change(
                    fn=toggle_api_enhancement,
                    inputs=api_toggle,
                    outputs=api_status
                )
            
            # Add Research Activity Monitor
            with gr.Accordion("📊 Research Activity Monitor", open=True):
                research_activity_display = gr.Markdown("**No active research**")
                research_log_display = gr.Textbox(
                    label="Research Event Log", 
                    lines=10, 
                    interactive=False,
                    value="Research events will appear here..."
                )
                
                with gr.Row():
                    refresh_log_btn = gr.Button("🔄 Refresh Log")
                    clear_log_btn = gr.Button("🗑️ Clear Log")
                
                # Quick research buttons
                gr.Markdown("### 🚀 Quick Research Topics")
                with gr.Row():
                    for topic_key, prompt in list(RESEARCH_PROMPTS.items())[:5]:
                        btn = gr.Button(topic_key.replace('_', ' ').title(), size="sm")
                        btn.click(
                            fn=lambda p=prompt: p,
                            outputs=chat_interface.textbox,
                            queue=False
                        )
                
                with gr.Row():
                    for topic_key, prompt in list(RESEARCH_PROMPTS.items())[5:]:
                        btn = gr.Button(topic_key.replace('_', ' ').title(), size="sm")
                        btn.click(
                            fn=lambda p=prompt: p,
                            outputs=chat_interface.textbox,
                            queue=False
                        )
                
                # Connect research monitor functions
                def update_research_activity():
                    """Update research activity display."""
                    if research_detector.is_research_active():
                        active = research_detector.get_active_research()
                        return f"""**🔬 RESEARCH ACTIVE**
Topic: {active['topic']}
Started: {active['start_time'].strftime('%H:%M:%S')}
Status: Running..."""
                    else:
                        return "**No active research**"
                
                def get_research_event_log():
                    """Get formatted research event log."""
                    events = research_detector.get_research_log()
                    if not events:
                        return "No research events yet..."
                    
                    log_text = ""
                    for event in reversed(events[-20:]):  # Show last 20 events
                        timestamp = event['timestamp'].split('T')[1].split('.')[0]  # Get time part
                        log_text += f"[{timestamp}] {event['message']}\n"
                    
                    return log_text
                
                def clear_research_log():
                    """Clear the research log."""
                    research_detector.research_log = []
                    return "Research log cleared"
                
                refresh_log_btn.click(
                    fn=lambda: (update_research_activity(), get_research_event_log()),
                    outputs=[research_activity_display, research_log_display]
                )
                
                clear_log_btn.click(
                    fn=clear_research_log,
                    outputs=research_log_display
                )
                
                # Auto-refresh research activity
                demo.load(
                    fn=lambda: (update_research_activity(), get_research_event_log()),
                    outputs=[research_activity_display, research_log_display],
                    every=2  # Update every 2 seconds
                )
            
            with gr.Tabs():
                with gr.Tab("📊 System Status"):
                    status_btn = gr.Button("Run Diagnostics")
                    status_output = gr.Textbox(lines=15, interactive=False)
                    status_btn.click(fn=run_system_test, outputs=status_output)
                
                if graphrag_available():
                    with gr.Tab("🔍 Graph Explorer"):
                        with gr.Row():
                            entity_input = gr.Textbox(placeholder="Entity name", label="Entity")
                            viz_btn = gr.Button("Visualize")
                        graph_output = gr.HTML()
                        viz_btn.click(
                            fn=lambda e: f"""
                            <div style='padding: 20px; text-align: center;'>
                                <h3>Graph for: {e}</h3>
                                <p>Interactive graph would appear here</p>
                                <p><em>Try asking the AI: \"Show me a graph of {e}\"</em></p>
                            </div>
                            """,
                            inputs=entity_input,
                            outputs=graph_output
                        )
                
                with gr.Tab("🔬 Research Lab"):
                    research_input = gr.Textbox(label="Research Goal", placeholder="What would you like to research?")
                    research_btn = gr.Button("Start Research")
                    research_output = gr.Markdown()
                    research_btn.click(fn=run_research_sync, inputs=research_input, outputs=research_output)
                
                with gr.Tab("🧬 Autonomous Research"):
                    gr.Markdown("""
                    ### AI Research Scientist Mode
                    
                    Start an autonomous research project where the AI conducts independent research cycles:
                    - Generates and tests hypotheses
                    - Reviews literature
                    - Runs simulations
                    - Designs experiments
                    - Proposes innovations
                    
                    The AI will work autonomously, logging progress hourly.
                    """)
                    
                    with gr.Row():
                        research_question = gr.Textbox(
                            label="Research Question",
                            placeholder="e.g., How can we improve muscle recovery after intense training?",
                            lines=2
                        )
                        project_name = gr.Textbox(
                            label="Project Name (optional)",
                            placeholder="e.g., muscle_recovery_study"
                        )
                        
                    with gr.Row():
                        start_research_btn = gr.Button("🚀 Start Autonomous Research", variant="primary")
                        pause_research_btn = gr.Button("⏸️ Pause Research")
                        resume_research_btn = gr.Button("▶️ Resume Research")
                        
                    research_status = gr.Markdown("**Status:** No active research")
                    
                    # Research log viewer
                    gr.Markdown("### 📊 Research Log")
                    with gr.Row():
                        refresh_log_btn = gr.Button("🔄 Refresh Log")
                        export_log_btn = gr.Button("📥 Export Research")
                        
                    research_log_display = gr.Markdown(
                        value="*Start a research project to see logs here*",
                        elem_id="research-log"
                    )
                    
                    # Connect autonomous research functions
                    start_research_btn.click(
                        fn=start_autonomous_research,
                        inputs=[research_question, project_name],
                        outputs=[research_status, research_log_display]
                    )
                    
                    pause_research_btn.click(
                        fn=pause_autonomous_research,
                        outputs=research_status
                    )
                    
                    resume_research_btn.click(
                        fn=resume_autonomous_research,
                        outputs=research_status
                    )
                    
                    refresh_log_btn.click(
                        fn=get_research_log,
                        outputs=research_log_display
                    )
                    
                    export_log_btn.click(
                        fn=export_research_project,
                        outputs=research_status
                    )
                    
                    # Auto-refresh research log every 30 seconds
                    demo.load(
                        fn=get_research_log,
                        outputs=research_log_display,
                        every=30
                    )
                
                with gr.Tab("📚 Research Library"):
                    gr.Markdown("""
                    ### Research Document Library
                    
                    Browse all research documents, papers, and findings organized by project.
                    """)
                    
                    library_stats = gr.Markdown()
                    
                    with gr.Row():
                        project_selector = gr.Dropdown(
                            label="Select Project",
                            choices=[],
                            interactive=True
                        )
                        refresh_library_btn = gr.Button("🔄 Refresh")
                        
                    project_documents = gr.Markdown()
                    
                    # Connect library functions
                    refresh_library_btn.click(
                        fn=refresh_research_library,
                        outputs=[library_stats, project_selector, project_documents]
                    )
                    
                    project_selector.change(
                        fn=get_project_documents,
                        inputs=project_selector,
                        outputs=project_documents
                    )
                    
                    # Load library on startup
                    demo.load(
                        fn=refresh_research_library,
                        outputs=[library_stats, project_selector, project_documents]
                    )
                
                with gr.Tab("📝 Training Log"):
                    log_output = gr.Textbox(label="training.log (tail)", lines=20, interactive=False)

                with gr.Tab("🧪 Experimental Research"):
                    gr.Markdown("""
                    ### Rapid Hypothesis Testing Engine
                    
                    Minimal overhead, maximum discovery. Give it a question and let it:
                    - Generate hypotheses
                    - Search your 11GB knowledge base
                    - Run simple simulations
                    - Generate innovations
                    
                    No red tape, just rapid experimentation.
                    """)
                    
                    with gr.Row():
                        research_question = gr.Textbox(
                            label="Research Question",
                            placeholder="e.g., How to maximize muscle protein synthesis?",
                            lines=2
                        )
                        max_iterations = gr.Slider(
                            minimum=1, maximum=20, value=5,
                            label="Max Iterations",
                            info="How many research cycles to run"
                        )
                        
                    with gr.Row():
                        start_experimental_btn = gr.Button("🚀 Start Experimental Research", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear")
                        
                    research_output = gr.Markdown(label="Research Progress")
                    
                    with gr.Accordion("📊 Research Summary", open=False):
                        summary_output = gr.JSON(label="Summary Data")
                        
                    # Connect experimental research functions
                    start_experimental_btn.click(
                        fn=run_experimental_research,
                        inputs=[research_question, max_iterations],
                        outputs=[research_output, summary_output]
                    )
                    
                    clear_btn.click(
                        fn=lambda: ("", None),
                        outputs=[research_output, summary_output]
                    )

            # Periodically update training log (last 200 lines)
            def tail_log():
                import os
                log_path = Path("training.log")
                if not log_path.exists():
                    return "training.log not found (start training to generate)"
                try:
                    with open(log_path, "r", errors="ignore") as fh:
                        lines = fh.readlines()[-200:]
                    return "".join(lines)
                except Exception as exc:
                    return f"Error reading log: {exc}"

            demo.load(tail_log, None, log_output, every=3)  # update every 3 s
        
        _last_complete_time = {"ts": 0.0}

        def update_progress_display():
            percentage, msg = get_progress()
            now = time.time()
            # Hide after 5 s when finished
            if percentage >= 100:
                if _last_complete_time["ts"] == 0:
                    _last_complete_time["ts"] = now
                elif now - _last_complete_time["ts"] > 5:
                    return gr.update(visible=False)
            else:
                _last_complete_time["ts"] = 0.0
            return gr.update(value=f"🔄 **Status:** {percentage:.1f}%\n\n{msg}" if msg else "", visible=bool(msg))
        
        demo.load(
            fn=update_progress_display,
            outputs=progress_display,
            every=2
        )
    
    # Launch the interface
    def is_port_available(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except OSError:
            return False
    
    ports_to_try = [7860, 7861, 7862, 7863, 7864, 7865]
    
    for port in ports_to_try:
        if is_port_available(port):
            print(f"\n🚀 **AI Assistant ready at http://localhost:{port}**")
            print("💬 Just start chatting - I understand natural language!")
            try:
                demo.launch(
                    share=False,
                    server_name="0.0.0.0",
                    server_port=port,
                    quiet=True,
                    show_error=True
                )
                return demo
            except OSError:
                continue
    
    print("\n❌ Could not find available port. Try: lsof -ti:7860 | xargs kill -9")

# Autonomous research functions

def start_autonomous_research(question: str, project_name: str = None) -> Tuple[str, str]:
    """Start an autonomous research project."""
    global research_agent, research_logger
    
    if not question:
        return "❌ Please provide a research question", ""
        
    try:
        # Initialize research agent if needed
        if research_agent is None:
            research_agent = AutonomousResearchAgent()
            research_logger = ResearchLogger(Path("research_projects/logs"))
            
        # Start research
        result = asyncio.run(research_agent.start_research(question, project_name))
        
        status = f"""
✅ **Research Started!**

**Project:** {result['project_name']}
**Question:** {question}
**Status:** Active

The AI is now conducting autonomous research. Check the log below for progress.
"""
        
        # Get initial log
        log = asyncio.run(get_research_log_async())
        
        return status, log
        
    except Exception as e:
        return f"❌ Error starting research: {str(e)}", ""
        
def pause_autonomous_research() -> str:
    """Pause the current research."""
    global research_agent
    
    if research_agent is None:
        return "**Status:** No active research"
        
    try:
        asyncio.run(research_agent.pause_research())
        return "**Status:** Research paused ⏸️"
    except Exception as e:
        return f"**Status:** Error - {str(e)}"
        
def resume_autonomous_research() -> str:
    """Resume paused research."""
    global research_agent
    
    if research_agent is None:
        return "**Status:** No active research"
        
    try:
        asyncio.run(research_agent.resume_research())
        return "**Status:** Research resumed ▶️"
    except Exception as e:
        return f"**Status:** Error - {str(e)}"
        
def get_research_log() -> str:
    """Get the current research log."""
    return asyncio.run(get_research_log_async())
    
async def get_research_log_async() -> str:
    """Async version of get_research_log."""
    global research_agent, research_logger
    
    if research_agent is None or research_agent.current_project is None:
        return "*No active research project*"
        
    try:
        project_name = research_agent.current_project['name']
        log_content = research_logger.get_latest_log_content(project_name, lines=100)
        
        # Add current status
        status = await research_agent.get_current_status()
        
        header = f"""
### 📊 Current Research Status

**Project:** {status['project']['name']}
**Phase:** {status['current_phase'] or 'Initializing'}
**Iteration:** {status['iteration']}
**Hypotheses:** {status['hypotheses_count']}
**Findings:** {status['findings_count']}

---

### 📝 Research Log

"""
        
        return header + log_content
        
    except Exception as e:
        return f"Error reading log: {str(e)}"
        
def export_research_project() -> str:
    """Export the current research project."""
    global research_agent, research_logger
    
    if research_agent is None or research_agent.current_project is None:
        return "**Status:** No active research to export"
        
    try:
        project_name = research_agent.current_project['name']
        
        # Run async export in sync context
        async def export():
            # Export logs
            log_path = await research_logger.export_logs(project_name, format='markdown')
            
            # Export library documents
            library_path = await research_agent.library.export_project(project_name)
            
            return log_path, library_path
            
        log_path, library_path = asyncio.run(export())
        
        return f"""
✅ **Research Exported!**

**Log:** {log_path}
**Documents:** {library_path}

The research has been exported to the above locations.
"""
        
    except Exception as e:
        return f"**Status:** Export error - {str(e)}"
        
def refresh_research_library() -> Tuple[str, gr.Dropdown, str]:
    """Refresh the research library view."""
    global research_agent
    
    if research_agent is None:
        research_agent = AutonomousResearchAgent()
        
    try:
        # Get library stats
        stats = research_agent.library.get_library_stats()
        
        stats_text = f"""
### 📊 Library Statistics

- **Total Documents:** {stats['total_documents']}
- **Total Projects:** {stats['total_projects']}
- **Total Size:** {stats['total_size_mb']} MB
- **Document Types:** {', '.join(f"{k}: {v}" for k, v in stats['documents_by_type'].items())}
"""
        
        # Get project list
        projects = research_agent.library.list_projects()
        project_names = [p['name'] for p in projects]
        
        # Create dropdown update
        dropdown = gr.Dropdown(choices=project_names, value=project_names[0] if project_names else None)
        
        # Get documents for first project
        if project_names:
            docs_text = asyncio.run(get_project_documents_async(project_names[0]))
        else:
            docs_text = "*No projects found*"
            
        return stats_text, dropdown, docs_text
        
    except Exception as e:
        return f"Error: {str(e)}", gr.Dropdown(choices=[]), ""
        
def get_project_documents(project_name: str) -> str:
    """Get documents for a specific project."""
    return asyncio.run(get_project_documents_async(project_name))
    
async def get_project_documents_async(project_name: str) -> str:
    """Async version of get_project_documents."""
    global research_agent
    
    if not project_name:
        return "*Select a project*"
        
    try:
        # Get project summary
        summary = await research_agent.library.get_project_summary(project_name)
        
        text = f"""
### 📁 Project: {project_name}

**Created:** {summary['created']}
**Documents:** {summary['total_documents']}
**Size:** {summary['total_size_mb']} MB

#### Document Types:
"""
        
        for doc_type, count in summary['document_types'].items():
            text += f"- {doc_type}: {count}\n"
            
        # Get recent documents
        docs = await research_agent.library.search_documents(project=project_name)
        
        text += "\n#### Recent Documents:\n"
        for doc in docs[:10]:  # Show last 10
            text += f"- {doc['id']}\n"
            
        return text
        
    except Exception as e:
        return f"Error loading project: {str(e)}"

def run_experimental_research(question: str, max_iterations: int) -> Tuple[str, Dict]:
    """Run experimental research and return results."""
    try:
        engine = ExperimentalResearchEngine()
        engine = integrate_with_graph(engine)  # Add graph capabilities if available
        
        # Run research loop asynchronously
        findings = asyncio.run(engine.research_loop(question, max_iterations))
        
        # Read the log file for display
        log_content = ""
        if engine.log_file.exists():
            with open(engine.log_file, 'r') as f:
                log_content = f.read()
                
        # Get summary
        summary = engine.get_summary()
        
        # Format output
        output = f"""# 🧪 Experimental Research Results

{log_content}

## 🎯 Top Innovations Generated:
"""
        for i, innovation in enumerate(summary['top_innovations'], 1):
            output += f"{i}. {innovation}\n"
            
        return output, summary
        
    except Exception as e:
        logger.error(f"Experimental research error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Experimental Research Error: {str(e)}", {}

if __name__ == "__main__":
    create_unified_gui() 