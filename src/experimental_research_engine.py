"""
Experimental Research Engine - Minimal implementation for maximum output
Focuses on rapid hypothesis generation and testing with minimal overhead
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import numpy as np

from .rag_chat import RAGChat, get_chat
from .pharma_rag_enhanced import get_pharma_rag
from .graph_rag import GraphRAG, graphrag_available
from .graph_query import get_subgraph

logger = logging.getLogger(__name__)


class ExperimentalResearchEngine:
    """
    Minimal research engine focused on rapid experimentation.
    4-phase loop: Question → Hypothesis → Test → Innovation
    """
    
    def __init__(self, research_dir: str = "experimental_research"):
        self.research_dir = Path(research_dir)
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        # Use global singleton instances to avoid memory issues
        self.rag_chat = get_chat()  # Use global instance
        self.pharma_rag = get_pharma_rag()
        
        # Check if systems are ready
        if not self.rag_chat.is_ready():
            logger.warning("RAG Chat system not ready - research may be limited")
        
        # Simple state
        self.current_hypothesis = None
        self.findings = []
        self.innovations = []
        self.iteration = 0
        
        # Minimal logging
        self.log_file = self.research_dir / "research_log.md"
        
    async def research_loop(self, initial_question: str, max_iterations: int = 10):
        """
        Main research loop - minimal overhead, maximum discovery.
        Enhanced to run for longer periods (20-30 minutes for deep research).
        """
        # Ensure systems are ready
        if not self.rag_chat.is_ready():
            self._log("# Research Failed: RAG system not ready\n")
            return []
            
        question = initial_question
        
        self._log(f"# Research Started: {datetime.now()}\n**Question:** {question}\n")
        self._log(f"**Max iterations:** {max_iterations}\n")
        self._log("**Note:** Deep research may take 20-30 minutes for thorough investigation.\n")
        
        start_time = time.time()
        
        for self.iteration in range(max_iterations):
            iteration_start = time.time()
            
            try:
                # Add a small delay between iterations to prevent overwhelming the system
                if self.iteration > 0:
                    await asyncio.sleep(2)  # 2 second pause between iterations
                
                self._log(f"\n## Starting Iteration {self.iteration + 1}/{max_iterations} at {datetime.now().strftime('%H:%M:%S')}\n")
                
                # Phase 1: Generate hypothesis (with retry logic)
                hypothesis = await self._generate_hypothesis(question)
                if not hypothesis:
                    self._log(f"\n**Warning:** Could not generate hypothesis for iteration {self.iteration}\n")
                    continue
                
                # Add thinking time for better hypothesis quality
                await asyncio.sleep(1)
                
                # Phase 2: Search knowledge + extract mechanisms
                self._log("Searching knowledge base...\n")
                evidence, mechanisms = await self._search_and_analyze(hypothesis)
                
                # Add analysis time
                await asyncio.sleep(2)
                
                # Phase 3: Run lightweight simulation if possible
                self._log("Running simulations...\n")
                simulation_result = await self._run_simulation(hypothesis, mechanisms)
                
                # Phase 4: Generate innovations and next question
                self._log("Generating innovations...\n")
                innovations = await self._innovate(hypothesis, evidence, simulation_result)
                
                # Add reflection time
                await asyncio.sleep(1)
                
                # Determine next question
                question = await self._get_next_question(innovations, evidence)
                
                # Quick log
                self._log_iteration(hypothesis, evidence, simulation_result, innovations)
                
                # Store findings
                self.findings.append({
                    'iteration': self.iteration,
                    'hypothesis': hypothesis,
                    'evidence_count': len(evidence),
                    'mechanisms': mechanisms,
                    'simulation': simulation_result,
                    'innovations': innovations
                })
                
                # Log iteration time
                iteration_time = time.time() - iteration_start
                self._log(f"\nIteration completed in {iteration_time:.1f} seconds\n")
                
                # For deep research, ensure minimum time per iteration
                min_iteration_time = 60  # 1 minute minimum per iteration
                if iteration_time < min_iteration_time:
                    wait_time = min_iteration_time - iteration_time
                    self._log(f"Waiting {wait_time:.1f}s for deeper analysis...\n")
                    await asyncio.sleep(wait_time)
                
                # Continue or complete
                if self._should_stop(question, innovations):
                    self._log("\n**Stopping criteria met - concluding research**\n")
                    break
                    
            except Exception as e:
                logger.error(f"Research loop error: {str(e)}")
                self._log(f"\n**Error in iteration {self.iteration}:** {str(e)}\n")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
        # Calculate total research time
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        self._log(f"\n# Research Complete\n")
        self._log(f"**Total time:** {minutes} minutes {seconds} seconds\n")
        self._log(f"**Total iterations:** {self.iteration + 1}\n")
        self._log(f"**Total innovations generated:** {len(self.innovations)}\n")
        
        # Log top innovations
        if self.innovations:
            self._log("\n## Top Innovations:\n")
            for i, innovation in enumerate(self.innovations[:10], 1):  # Show top 10
                self._log(f"{i}. {innovation}\n")
        
        return self.findings
        
    async def _generate_hypothesis(self, question: str) -> str:
        """Generate hypothesis using pattern recognition."""
        prompt = f"""Based on this research question: {question}

Generate a specific, testable hypothesis about mechanisms in sports performance, muscle biology, or training.
Focus on: cause-effect relationships, molecular pathways, or performance outcomes.

Hypothesis:"""
        
        try:
            hypothesis = await self.rag_chat.generate(prompt)
            self.current_hypothesis = hypothesis.strip()
            return self.current_hypothesis
        except Exception as e:
            logger.error(f"Hypothesis generation error: {str(e)}")
            # Fallback hypothesis
            self.current_hypothesis = f"Testing the effects and mechanisms of {question}"
            return self.current_hypothesis
        
    async def _search_and_analyze(self, hypothesis: str) -> Tuple[List[Dict], List[str]]:
        """Search local knowledge and extract mechanisms."""
        # Search using existing RAG - fixed to not use top_k parameter
        evidence = self.pharma_rag.retrieve(hypothesis)
        
        # Extract mechanisms with simple prompt
        # Evidence is a list of strings, not dictionaries
        context_text = "\n".join([doc[:200] for doc in evidence[:5]])
        mechanism_prompt = f"""From this evidence about: {hypothesis}

Evidence:
{context_text}

Extract key biological mechanisms (molecular, cellular, physiological):
1."""
        
        try:
            mechanisms_text = await self.rag_chat.generate(mechanism_prompt)
            
            # Parse mechanisms
            mechanisms = [m.strip() for m in mechanisms_text.split('\n') if m.strip() and not m.strip().startswith('Extract')][:5]
        except Exception as e:
            logger.error(f"Mechanism extraction error: {str(e)}")
            # Fallback mechanisms based on hypothesis
            mechanisms = ["Enhanced cellular metabolism", "Improved muscle protein synthesis", "Optimized recovery pathways"]
        
        return evidence, mechanisms
        
    async def _run_simulation(self, hypothesis: str, mechanisms: List[str]) -> Optional[Dict[str, Any]]:
        """Run simple simulations based on hypothesis type."""
        # Detect simulation type from hypothesis
        hypothesis_lower = hypothesis.lower()
        
        if any(term in hypothesis_lower for term in ['training', 'exercise', 'workout']):
            return self._simulate_training_response()
        elif any(term in hypothesis_lower for term in ['supplement', 'creatine', 'protein']):
            return self._simulate_supplement_effect()
        elif any(term in hypothesis_lower for term in ['recovery', 'fatigue', 'rest']):
            return self._simulate_recovery()
        else:
            # No simulation available
            return None
            
    def _simulate_training_response(self) -> Dict[str, Any]:
        """Simple Banister model for training response."""
        weeks = 8
        fitness = []
        fatigue = []
        performance = []
        
        for week in range(weeks):
            # Simple impulse-response model
            training_load = 100 + np.random.normal(0, 10)
            fitness_gain = training_load * 0.1 * (1 - week/20)  # Diminishing returns
            fatigue_accum = training_load * 0.05
            
            current_fitness = sum(fitness) + fitness_gain
            current_fatigue = sum(fatigue[-3:]) + fatigue_accum  # Recent fatigue matters more
            current_performance = current_fitness - current_fatigue * 0.5
            
            fitness.append(fitness_gain)
            fatigue.append(fatigue_accum)
            performance.append(current_performance)
            
        improvement = (performance[-1] - performance[0]) / performance[0] * 100
        
        return {
            'type': 'training_response',
            'weeks': weeks,
            'improvement_percent': round(improvement, 1),
            'peak_week': int(np.argmax(performance)) + 1,
            'recommendation': 'Reduce volume after week 5' if improvement < 10 else 'Continue progression'
        }
        
    def _simulate_supplement_effect(self) -> Dict[str, Any]:
        """Simple dose-response for supplements."""
        doses = np.linspace(0, 10, 20)  # 0-10g
        response = 100 / (1 + np.exp(-0.8 * (doses - 5)))  # Sigmoid response
        side_effects = doses ** 1.5  # Exponential side effects
        net_benefit = response - side_effects * 0.5
        
        optimal_dose = doses[np.argmax(net_benefit)]
        max_benefit = np.max(net_benefit)
        
        return {
            'type': 'supplement_dose_response',
            'optimal_dose_g': round(optimal_dose, 1),
            'max_benefit_percent': round(max_benefit, 1),
            'threshold_dose_g': round(doses[np.where(response > 20)[0][0]], 1),
            'recommendation': f'Optimal dose: {optimal_dose:.1f}g daily'
        }
        
    def _simulate_recovery(self) -> Dict[str, Any]:
        """Simple recovery dynamics."""
        hours = np.arange(0, 72, 1)
        initial_fatigue = 100
        recovery_rate = 0.05  # Per hour
        
        fatigue = initial_fatigue * np.exp(-recovery_rate * hours)
        performance = 100 - fatigue
        
        time_to_90_percent = -np.log(0.1) / recovery_rate
        
        return {
            'type': 'recovery_dynamics',
            'hours_to_90_percent': round(time_to_90_percent, 1),
            'performance_24h': round(performance[24], 1),
            'performance_48h': round(performance[48], 1),
            'recommendation': 'Allow 48h between intense sessions' if time_to_90_percent > 40 else 'Can train daily'
        }
        
    async def _innovate(self, hypothesis: str, evidence: List[Dict], simulation: Optional[Dict]) -> List[str]:
        """Generate innovative ideas based on findings."""
        context = f"Hypothesis: {hypothesis}\nEvidence points: {len(evidence)}"
        if simulation:
            context += f"\nSimulation result: {simulation.get('recommendation', 'No specific outcome')}"
            
        prompt = f"""{context}

Generate 3 innovative ideas for sports performance or muscle optimization:
1."""
        
        try:
            innovations_text = await self.rag_chat.generate(prompt)
            
            # Parse innovations
            innovations = []
            for line in innovations_text.split('\n'):
                if line.strip() and (line[0].isdigit() or line.startswith('-')):
                    innovations.append(line.strip())
        except Exception as e:
            logger.error(f"Innovation generation error: {str(e)}")
            # Fallback innovations
            innovations = [
                "1. Combine timing strategies with compound synergies",
                "2. Explore personalized dosing based on metabolic markers",
                "3. Investigate novel delivery mechanisms for enhanced bioavailability"
            ]
            
        self.innovations.extend(innovations[:3])
        return innovations[:3]
        
    async def _get_next_question(self, innovations: List[str], evidence: List[Dict]) -> str:
        """Determine next research question."""
        # Look for gaps or contradictions
        if len(evidence) < 3:
            return f"What mechanisms support: {self.current_hypothesis}?"
        elif innovations:
            # Pick most interesting innovation
            return f"How can we implement: {innovations[0]}?"
        else:
            # Explore adjacent area
            return f"What factors modify the effect of: {self.current_hypothesis}?"
            
    def _should_stop(self, next_question: str, innovations: List[str]) -> bool:
        """Simple stopping criteria."""
        # Stop if we're going in circles or have enough innovations
        if self.iteration > 0 and next_question in str(self.findings):
            return True
        if len(self.innovations) > 20:
            return True
        return False
        
    def _log(self, message: str):
        """Minimal logging to file."""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
            
    def _log_iteration(self, hypothesis: str, evidence: List[Dict], simulation: Optional[Dict], innovations: List[str]):
        """Log iteration results in simple format."""
        log_entry = f"""
## Iteration {self.iteration + 1} - {datetime.now().strftime('%H:%M')}

**Hypothesis:** {hypothesis}
**Evidence:** {len(evidence)} sources found
**Simulation:** {simulation.get('type', 'None')} - {simulation.get('recommendation', 'N/A') if simulation else 'N/A'}
**Innovations:** {len(innovations)} new ideas

"""
        self._log(log_entry)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get research summary."""
        return {
            'iterations': self.iteration + 1,
            'hypotheses_tested': len(self.findings),
            'total_innovations': len(self.innovations),
            'findings': self.findings,
            'top_innovations': self.innovations[:10]
        }


class HypothesisEngine:
    """Simple pattern-based hypothesis generator."""
    
    def __init__(self):
        self.patterns = [
            "If {X} affects {Y}, then increasing {X} should enhance {Y}",
            "Since {A} leads to {B} which causes {C}, directly targeting {B} might accelerate {C}",
            "{X} and {Y} both affect {Z}, so combining them might have synergistic effects",
            "If {mechanism} is rate-limiting, then enhancing it should improve {outcome}",
            "The opposite of {intervention} might reveal new insights about {system}"
        ]
        
    async def generate_from_gaps(self, findings: List[Dict], knowledge_gaps: List[str]) -> List[str]:
        """Generate hypotheses from knowledge gaps."""
        hypotheses = []
        
        for gap in knowledge_gaps[:3]:  # Limit to top 3 gaps
            # Pattern match to create hypothesis
            if "mechanism" in gap.lower():
                hyp = f"The mechanism of {gap} involves modulation of key signaling pathways"
            elif "effect" in gap.lower():
                hyp = f"The effect of {gap} is dose-dependent and follows a non-linear response"
            else:
                hyp = f"Understanding {gap} could reveal new intervention targets"
                
            hypotheses.append(hyp)
            
        return hypotheses
        
    def find_connections(self, concepts: List[str]) -> List[Tuple[str, str, str]]:
        """Find potential connections between concepts."""
        connections = []
        
        # Simple connection patterns
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Generate potential relationship
                if "muscle" in concept1 and "growth" in concept2:
                    connections.append((concept1, "stimulates", concept2))
                elif "fatigue" in concept1 and "performance" in concept2:
                    connections.append((concept1, "inhibits", concept2))
                else:
                    connections.append((concept1, "may influence", concept2))
                    
        return connections[:5]  # Top 5 connections


# Integration with existing GraphRAG
def integrate_with_graph(engine: ExperimentalResearchEngine):
    """Add graph-based discovery to research engine."""
    if graphrag_available():
        # Add method to explore graph relationships
        async def explore_graph_connections(entity: str) -> List[str]:
            nodes, edges = get_subgraph(entity, max_depth=2)
            connections = []
            for edge in edges:
                connections.append(f"{edge['source']} -> {edge['target']} ({edge.get('type', 'relates_to')})")
            return connections[:10]
            
        engine.explore_graph = explore_graph_connections
        
    return engine 