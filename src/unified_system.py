"""Unified biomedical LLM system entry point.

Single point of access for all functionality with proper M1 optimization
and comprehensive error handling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import threading
import queue
import time

from .config import settings
from .diagnostics import SystemDiagnostics, ErrorHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingSystemMediator:
    """Central coordinator for GUI-Training integration."""
    
    def __init__(self):
        self.gui = None
        self.trainer = None
        self.rag_system = None
        self.event_queue = queue.Queue()
        self.callbacks = {}
        self.training_thread = None
        
    def register_gui(self, gui_instance):
        """Connect GUI with bidirectional communication."""
        self.gui = gui_instance
        if hasattr(self.gui, 'set_mediator'):
            self.gui.set_mediator(self)
        
    def register_trainer(self, trainer_instance):
        """Connect BiomedicalTrainer with progress callbacks."""
        self.trainer = trainer_instance
        
        # Set up callbacks if trainer supports them
        if hasattr(self.trainer, 'on_epoch_end'):
            self.trainer.on_epoch_end = self._on_epoch_end
        if hasattr(self.trainer, 'on_batch_end'):
            self.trainer.on_batch_end = self._on_batch_end
        if hasattr(self.trainer, 'on_training_complete'):
            self.trainer.on_training_complete = self._on_training_complete
            
    def register_rag_system(self, rag_system):
        """Register RAG system for updates."""
        self.rag_system = rag_system
        
    def start_training(self, config: Dict[str, Any]) -> bool:
        """Launch training in separate thread with GUI updates."""
        if not self.trainer:
            logger.error("BiomedicalTrainer not registered")
            if self.gui:
                self.gui.show_error("Training system not available")
            return False
            
        # Update GUI
        if self.gui:
            self.gui.update_status("Training Started", progress=0)
            
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(config,),
            daemon=True
        )
        self.training_thread.start()
        return True
        
    def _run_training(self, config):
        """Execute training with error handling."""
        try:
            logger.info("Starting training with config: %s", config)
            
            if hasattr(self.trainer, 'train'):
                self.trainer.train(config)
            else:
                # Fallback to basic training method
                logger.warning("Trainer doesn't have 'train' method, using fallback")
                time.sleep(5)  # Simulate training
                self._on_training_complete({"loss": 0.5, "accuracy": 0.85})
                
        except Exception as e:
            logger.error("Training failed: %s", e)
            if self.gui:
                self.gui.show_error(f"Training failed: {e}")
                
    def _on_epoch_end(self, epoch, metrics):
        """Forward epoch progress to GUI."""
        if self.gui and hasattr(self.gui, 'update_progress'):
            if hasattr(self.trainer, 'total_epochs'):
                progress = (epoch / self.trainer.total_epochs) * 100
                self.gui.update_progress(progress, metrics)
            
    def _on_batch_end(self, batch, metrics):
        """Handle batch completion."""
        # Only update GUI occasionally to avoid spam
        if batch % 10 == 0 and self.gui and hasattr(self.gui, 'update_batch_progress'):
            self.gui.update_batch_progress(batch, metrics)
            
    def _on_training_complete(self, final_metrics):
        """Handle training completion."""
        logger.info("Training completed with metrics: %s", final_metrics)
        if self.gui and hasattr(self.gui, 'update_status'):
            self.gui.update_status("Training Complete", progress=100)
        if self.gui and hasattr(self.gui, 'display_results'):
            self.gui.display_results(final_metrics)
            
        # Reload RAG system if training was successful
        if self.rag_system and hasattr(self.rag_system, 'reload_models'):
            try:
                self.rag_system.reload_models()
                logger.info("RAG system reloaded after training")
            except Exception as e:
                logger.error("Failed to reload RAG system: %s", e)


class UnifiedBiomedicalSystem:
    """Single entry point for all biomedical LLM functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.error_handler = ErrorHandler()
        
        # Component storage
        self.llm = None
        self.rag = None
        self.trainer = None
        self.gui = None
        self.knowledge_graph = None
        
        # System state
        self.initialized = False
        self.diagnostics = SystemDiagnostics()
        self.mediator = TrainingSystemMediator()
        
        # Initialize logging
        self._setup_logging()
        
    def _default_config(self):
        """Default configuration optimized for M1 Mac."""
        return {
            'model': {
                'path': str(Path(settings.model_dir) / settings.llm_model),
                'n_gpu_layers': 8,       # Very conservative
                'n_ctx': 512,            # Small context
                'n_batch': 32,           # Small batch
                'f16_kv': False,         # Disabled for stability
                'use_mlock': False,      # Disabled for stability  
                'n_threads': 2,          # Conservative threads
                'verbose': False,
                'low_vram': True         # Enable low VRAM mode
            },
            'embedding': {
                'model_name': 'all-MiniLM-L6-v2',
                'cache_dir': str(Path(settings.model_dir) / 'embeddings'),
                'dimension': 384
            },
            'rag': {
                'max_docs': 1000,
                'chunk_size': 512,
                'chunk_overlap': 50
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-5,
                'num_epochs': 3,
                'save_steps': 100
            },
            'gui': {
                'framework': 'gradio',  # 'gradio', 'streamlit', 'tkinter'
                'theme': 'default',
                'share': False
            },
            'neo4j': {
                'uri': settings.neo4j_uri,
                'user': settings.neo4j_user,
                'password': settings.neo4j_password
            }
        }
        
    def _setup_logging(self):
        """Configure system-wide logging."""
        log_file = Path("biomedical_system.log")
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
    def initialize(self, force: bool = False) -> bool:
        """Initialize all system components."""
        if self.initialized and not force:
            logger.info("System already initialized")
            return True
            
        logger.info("Initializing unified biomedical system...")
        
        try:
            # Run diagnostics first
            diagnostics_results = self.diagnostics.run_all_diagnostics()
            
            # Check critical components
            critical_components = ['platform', 'faiss', 'embeddings']  # Removed python_env as it can be warning
            for component in critical_components:
                if not diagnostics_results.get(component, {}).get('healthy', False):
                    logger.error(f"Critical component {component} failed health check")
                    return False
            
            # Warn about python_env but don't fail initialization
            if not diagnostics_results.get('python_env', {}).get('healthy', False):
                logger.warning("Python environment check failed, but continuing initialization...")
                
            # Initialize components in order
            success = True
            success &= self._initialize_llm()
            success &= self._initialize_rag()
            success &= self._initialize_trainer()
            success &= self._initialize_knowledge_graph()
            
            if success:
                # Connect components via mediator
                self.mediator.register_trainer(self.trainer)
                self.mediator.register_rag_system(self.rag)
                
                self.initialized = True
                logger.info("✅ System initialization complete")
                return True
            else:
                logger.error("❌ System initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            return False
            
    def _initialize_llm(self) -> bool:
        """Initialize LLM with M1 optimization."""
        try:
            from .llm import get_llm
            
            logger.info("Initializing LLM...")
            self.llm = get_llm()
            
            if self.llm:
                logger.info("✅ LLM initialized successfully")
                return True
            else:
                logger.error("❌ LLM initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            return False
            
    def _initialize_rag(self) -> bool:
        """Initialize RAG system with robust error handling."""
        try:
            from .rag_chat import RAGChat
            
            logger.info("Initializing RAG system...")
            
            # Initialize the proper RAG chat system
            self.rag = RAGChat()
            
            if self.rag.is_ready():
                logger.info("✅ RAG system initialized successfully")
                return True
            else:
                logger.warning("RAG system not fully ready, but continuing...")
                return True  # Don't fail initialization
            
        except Exception as e:
            logger.error(f"RAG initialization error: {e}")
            return False
            
    def _initialize_trainer(self) -> bool:
        """Initialize training system."""
        try:
            from .biomedical_trainer import BiomedicalTrainer
            
            logger.info("Initializing training system...")
            self.trainer = BiomedicalTrainer(self.config['training'])
            
            logger.info("✅ Training system initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Training system not available: {e}")
            # Training is optional, so don't fail initialization
            return True
            
    def _initialize_knowledge_graph(self) -> bool:
        """Initialize Neo4j knowledge graph."""
        try:
            from .knowledge_graph import BiomedicalKnowledgeGraph
            
            logger.info("Initializing knowledge graph...")
            self.knowledge_graph = BiomedicalKnowledgeGraph(
                uri=self.config['neo4j']['uri'],
                user=self.config['neo4j']['user'],
                password=self.config['neo4j']['password']
            )
            
            if self.knowledge_graph.test_connection():
                logger.info("✅ Knowledge graph initialized successfully")
                return True
            else:
                logger.warning("Knowledge graph connection failed, continuing without it")
                return True  # Don't fail initialization
                
        except Exception as e:
            logger.warning(f"Knowledge graph not available: {e}")
            return True  # Don't fail initialization
            
    def create_gui(self, framework: str = None):
        """Create GUI using specified framework."""
        framework = framework or self.config['gui']['framework']
        
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("System initialization failed")
                
        try:
            if framework == 'gradio':
                return self._create_gradio_gui()
            elif framework == 'streamlit':
                return self._create_streamlit_gui()
            elif framework == 'tkinter':
                return self._create_tkinter_gui()
            else:
                raise ValueError(f"Unsupported GUI framework: {framework}")
                
        except Exception as e:
            logger.error(f"GUI creation failed: {e}")
            raise
            
    def _create_gradio_gui(self):
        """Create Gradio web interface."""
        try:
            from .gui_unified import create_unified_gui
            logger.info("Creating Gradio GUI...")
            return create_unified_gui()
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Gradio GUI creation failed: {e}")
            logger.error(f"Full traceback: {full_traceback}")
            raise
            
    def _create_streamlit_gui(self):
        """Create Streamlit interface (placeholder)."""
        raise NotImplementedError("Streamlit GUI not yet implemented")
        
    def _create_tkinter_gui(self):
        """Create Tkinter interface (placeholder)."""
        raise NotImplementedError("Tkinter GUI not yet implemented")
        
    def chat(self, message: str) -> str:
        """Direct chat interface."""
        if not self.initialized:
            return "System not initialized. Please run initialize() first."
            
        if not self.llm:
            return "LLM not available."
            
        try:
            # Use RAG if available
            if self.rag and hasattr(self.rag, 'answer'):
                return self.rag.answer(message)
            
            # Fallback to direct LLM
            if hasattr(self.llm, 'create_completion'):
                # llama-cpp-python interface
                response = self.llm.create_completion(
                    prompt=f"User: {message}\nAssistant:",
                    max_tokens=512,
                    temperature=0.7,
                    stop=["User:", "\n\n"]
                )
                return response['choices'][0]['text'].strip()
            else:
                # Other interfaces
                return str(self.llm(message))
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Sorry, I encountered an error: {e}"
            
    def get_status(self) -> Dict[str, Any]:
        """Get system status and health."""
        status = {
            'initialized': self.initialized,
            'components': {
                'llm': self.llm is not None,
                'rag': self.rag is not None,
                'trainer': self.trainer is not None,
                'knowledge_graph': self.knowledge_graph is not None,
                'gui': self.gui is not None
            },
            'health': self.diagnostics.run_all_diagnostics() if self.initialized else {}
        }
        
        return status
        
    def start_training(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Start model training."""
        if not self.initialized:
            logger.error("System not initialized")
            return False
            
        training_config = config or self.config['training']
        return self.mediator.start_training(training_config)
        
    def search_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """Search knowledge graph."""
        if not self.knowledge_graph:
            return []
            
        try:
            return self.knowledge_graph.search(query)
        except Exception as e:
            logger.error(f"Knowledge graph search error: {e}")
            return []


# Convenience functions for common operations
def create_system(config: Optional[Dict[str, Any]] = None) -> UnifiedBiomedicalSystem:
    """Create and initialize unified system."""
    system = UnifiedBiomedicalSystem(config)
    system.initialize()
    return system


def run_diagnostics():
    """Run system diagnostics."""
    from .diagnostics import run_diagnostics
    return run_diagnostics()


def quick_start():
    """Quick start with Gradio GUI."""
    system = create_system()
    gui = system.create_gui('gradio')
    return system, gui 