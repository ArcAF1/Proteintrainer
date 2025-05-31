"""
Autonomous Research Scientist Agent
Implements the iterative research loop for independent biomedical research
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging

from .pharma_rag_enhanced import get_pharma_rag
from .knowledge_gap_analyzer import KnowledgeGapAnalyzer
from .hypothesis_engine import HypothesisEngine
from .research_logger import ResearchLogger
from .research_library import ResearchLibrary
from .simulation_tools import SimulationEngine
from .rag_chat import RAGChat

logger = logging.getLogger(__name__)


class ResearchPhase(Enum):
    """Phases of the autonomous research loop."""
    PROBLEM_DEFINITION = "problem_definition"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_REFINEMENT = "hypothesis_refinement"
    EXPERIMENT_DESIGN = "experiment_design"
    CLINICAL_TRIAL_DESIGN = "clinical_trial_design"
    INNOVATION_PROPOSAL = "innovation_proposal"
    LOGGING_PLANNING = "logging_planning"


class ResearchStatus(Enum):
    """Status of the research project."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AutonomousResearchAgent:
    """
    Autonomous research scientist that conducts independent research cycles.
    Follows the methodology of a high-level medical/pharma R&D scientist.
    """
    
    def __init__(self, research_dir: str = "research_projects"):
        self.research_dir = Path(research_dir)
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pharma_rag = get_pharma_rag()
        self.gap_analyzer = KnowledgeGapAnalyzer()
        self.hypothesis_engine = HypothesisEngine()
        self.logger = ResearchLogger(self.research_dir / "logs")
        self.library = ResearchLibrary(self.research_dir / "library")
        self.simulation_engine = SimulationEngine()
        self.chat = RAGChat()
        
        # Research state
        self.current_project = None
        self.current_phase = None
        self.status = ResearchStatus.ACTIVE
        self.iteration_count = 0
        self.last_log_time = time.time()
        
        # Research memory
        self.hypotheses = []
        self.findings = []
        self.experiments = []
        self.innovations = []
        
    async def start_research(self, initial_question: str, project_name: str = None) -> Dict[str, Any]:
        """
        Start a new autonomous research project.
        
        Args:
            initial_question: The research question or problem to investigate
            project_name: Optional name for the project
            
        Returns:
            Project initialization details
        """
        # Create project
        if not project_name:
            project_name = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_project = {
            'name': project_name,
            'question': initial_question,
            'start_time': datetime.now(),
            'status': ResearchStatus.ACTIVE.value,
            'phases_completed': [],
            'current_iteration': 1
        }
        
        # Initialize project directory
        project_dir = self.research_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Log project start
        await self.logger.log_event(
            project_name,
            "PROJECT_START",
            {
                'initial_question': initial_question,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Start research loop
        asyncio.create_task(self._run_research_loop())
        
        return {
            'project_name': project_name,
            'status': 'started',
            'initial_question': initial_question
        }
        
    async def _run_research_loop(self):
        """
        Main autonomous research loop.
        Cycles through research phases iteratively.
        """
        while self.status == ResearchStatus.ACTIVE:
            try:
                # Check if hourly log is needed
                await self._check_hourly_log()
                
                # Execute current research phase
                phase_result = await self._execute_current_phase()
                
                # Determine next phase based on results
                next_phase = self._determine_next_phase(phase_result)
                
                # Update phase
                self.current_phase = next_phase
                
                # Check completion conditions
                if self._should_complete_research():
                    await self._complete_research()
                    break
                    
                # Brief pause between phases
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Research loop error: {str(e)}")
                await self.logger.log_error(
                    self.current_project['name'],
                    str(e),
                    self.current_phase.value if self.current_phase else "unknown"
                )
                
                # Continue research despite errors
                await asyncio.sleep(5)
                
    async def _execute_current_phase(self) -> Dict[str, Any]:
        """Execute the current research phase."""
        if not self.current_phase:
            self.current_phase = ResearchPhase.PROBLEM_DEFINITION
            
        phase_methods = {
            ResearchPhase.PROBLEM_DEFINITION: self._phase_problem_definition,
            ResearchPhase.LITERATURE_REVIEW: self._phase_literature_review,
            ResearchPhase.HYPOTHESIS_REFINEMENT: self._phase_hypothesis_refinement,
            ResearchPhase.EXPERIMENT_DESIGN: self._phase_experiment_design,
            ResearchPhase.CLINICAL_TRIAL_DESIGN: self._phase_clinical_trial_design,
            ResearchPhase.INNOVATION_PROPOSAL: self._phase_innovation_proposal,
            ResearchPhase.LOGGING_PLANNING: self._phase_logging_planning
        }
        
        method = phase_methods.get(self.current_phase)
        if method:
            return await method()
        else:
            logger.error(f"Unknown phase: {self.current_phase}")
            return {'status': 'error', 'phase': self.current_phase}
            
    async def _phase_problem_definition(self) -> Dict[str, Any]:
        """Phase 1: Define problem and generate initial hypothesis."""
        project_name = self.current_project['name']
        question = self.current_project['question']
        
        await self.logger.log_phase_start(project_name, "PROBLEM_DEFINITION", {
            'question': question,
            'iteration': self.iteration_count
        })
        
        # Analyze the question
        prompt = f"""
        As a biomedical research scientist specializing in sports, fitness, and human performance,
        analyze this research question and generate initial hypotheses:
        
        Question: {question}
        
        Provide:
        1. Clear problem definition
        2. 2-3 initial hypotheses with mechanistic rationale
        3. Key areas to investigate
        4. Potential impact if solved
        
        Focus on muscle biology, training, and performance enhancement.
        """
        
        response = await self.chat.generate(prompt)
        
        # Parse response and extract hypotheses
        hypotheses = self._extract_hypotheses(response)
        self.hypotheses.extend(hypotheses)
        
        # Save to library
        await self.library.save_document(
            project_name,
            "problem_definition",
            {
                'question': question,
                'analysis': response,
                'hypotheses': hypotheses,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        await self.logger.log_phase_complete(project_name, "PROBLEM_DEFINITION", {
            'hypotheses_generated': len(hypotheses)
        })
        
        return {
            'status': 'completed',
            'hypotheses': hypotheses,
            'next_phase': ResearchPhase.LITERATURE_REVIEW
        }
        
    async def _phase_literature_review(self) -> Dict[str, Any]:
        """Phase 2: Conduct mechanistic literature review."""
        project_name = self.current_project['name']
        current_hypothesis = self.hypotheses[0] if self.hypotheses else None
        
        if not current_hypothesis:
            return {'status': 'error', 'message': 'No hypothesis to investigate'}
            
        await self.logger.log_phase_start(project_name, "LITERATURE_REVIEW", {
            'hypothesis': current_hypothesis,
            'sources': 'local_first'
        })
        
        # Search local knowledge first
        local_docs = await self._search_local_knowledge(current_hypothesis)
        
        # Analyze mechanisms
        mechanism_analysis = await self._analyze_mechanisms(
            current_hypothesis, 
            local_docs
        )
        
        # Check knowledge gaps
        gaps = self.gap_analyzer.analyze_query_coverage(
            current_hypothesis,
            [d['content'] for d in local_docs]
        )
        
        # If significant gaps, search external sources
        external_docs = []
        if gaps['coverage_score'] < 0.6:
            external_docs = await self._search_external_sources(
                current_hypothesis,
                gaps['missing_aspects']
            )
            
        # Synthesize findings
        synthesis = await self._synthesize_literature(
            current_hypothesis,
            local_docs + external_docs,
            mechanism_analysis
        )
        
        # Save findings
        findings = {
            'hypothesis': current_hypothesis,
            'local_sources': len(local_docs),
            'external_sources': len(external_docs),
            'mechanisms': mechanism_analysis,
            'synthesis': synthesis,
            'knowledge_gaps': gaps,
            'timestamp': datetime.now().isoformat()
        }
        
        self.findings.append(findings)
        await self.library.save_document(
            project_name,
            f"literature_review_iter{self.iteration_count}",
            findings
        )
        
        await self.logger.log_phase_complete(project_name, "LITERATURE_REVIEW", {
            'documents_reviewed': len(local_docs) + len(external_docs),
            'coverage_score': gaps['coverage_score']
        })
        
        return {
            'status': 'completed',
            'findings': findings,
            'next_phase': ResearchPhase.HYPOTHESIS_REFINEMENT
        }
        
    async def _phase_hypothesis_refinement(self) -> Dict[str, Any]:
        """Phase 3: Refine hypothesis based on evidence."""
        project_name = self.current_project['name']
        
        await self.logger.log_phase_start(project_name, "HYPOTHESIS_REFINEMENT", {
            'current_hypotheses': len(self.hypotheses)
        })
        
        # Evaluate current hypothesis against findings
        if self.findings:
            latest_findings = self.findings[-1]
            evaluation = await self._evaluate_hypothesis(
                self.hypotheses[0],
                latest_findings
            )
            
            # Refine or generate new hypothesis
            if evaluation['support_level'] < 0.5:
                # Generate alternative hypothesis
                new_hypothesis = await self._generate_alternative_hypothesis(
                    self.hypotheses[0],
                    latest_findings
                )
                self.hypotheses.insert(0, new_hypothesis)
                
                result = {
                    'action': 'new_hypothesis',
                    'hypothesis': new_hypothesis,
                    'reason': evaluation['reason']
                }
            else:
                # Refine existing hypothesis
                refined = await self._refine_hypothesis(
                    self.hypotheses[0],
                    latest_findings
                )
                self.hypotheses[0] = refined
                
                result = {
                    'action': 'refined',
                    'hypothesis': refined,
                    'support_level': evaluation['support_level']
                }
        else:
            result = {'action': 'no_change', 'reason': 'No findings yet'}
            
        await self.library.save_document(
            project_name,
            f"hypothesis_refinement_iter{self.iteration_count}",
            result
        )
        
        await self.logger.log_phase_complete(
            project_name, 
            "HYPOTHESIS_REFINEMENT",
            result
        )
        
        # Decide next phase
        if result['action'] == 'new_hypothesis':
            next_phase = ResearchPhase.LITERATURE_REVIEW  # Loop back
        else:
            next_phase = ResearchPhase.EXPERIMENT_DESIGN
            
        return {
            'status': 'completed',
            'result': result,
            'next_phase': next_phase
        }
        
    async def _phase_experiment_design(self) -> Dict[str, Any]:
        """Phase 4: Design experiments and run simulations."""
        project_name = self.current_project['name']
        hypothesis = self.hypotheses[0]
        
        await self.logger.log_phase_start(project_name, "EXPERIMENT_DESIGN", {
            'hypothesis': hypothesis,
            'simulation_available': self.simulation_engine.available
        })
        
        # Design experiment
        experiment_design = await self._design_experiment(hypothesis)
        
        # Run simulation if possible
        simulation_results = None
        if self.simulation_engine.available and experiment_design.get('simulatable'):
            simulation_results = await self._run_simulation(experiment_design)
            
        # Analyze results
        analysis = await self._analyze_experiment_results(
            experiment_design,
            simulation_results
        )
        
        experiment = {
            'hypothesis': hypothesis,
            'design': experiment_design,
            'simulation_results': simulation_results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experiments.append(experiment)
        await self.library.save_document(
            project_name,
            f"experiment_{self.iteration_count}",
            experiment
        )
        
        await self.logger.log_phase_complete(project_name, "EXPERIMENT_DESIGN", {
            'simulated': simulation_results is not None,
            'outcome': analysis.get('outcome', 'unknown')
        })
        
        # Determine next phase
        if analysis.get('hypothesis_supported', False):
            next_phase = ResearchPhase.CLINICAL_TRIAL_DESIGN
        else:
            next_phase = ResearchPhase.HYPOTHESIS_REFINEMENT  # Loop back
            
        return {
            'status': 'completed',
            'experiment': experiment,
            'next_phase': next_phase
        }
        
    async def _phase_clinical_trial_design(self) -> Dict[str, Any]:
        """Phase 5: Design clinical trial or implementation study."""
        project_name = self.current_project['name']
        hypothesis = self.hypotheses[0]
        
        await self.logger.log_phase_start(project_name, "CLINICAL_TRIAL_DESIGN", {
            'hypothesis': hypothesis
        })
        
        # Design trial based on hypothesis and experiments
        trial_design = await self._design_clinical_trial(
            hypothesis,
            self.experiments[-1] if self.experiments else None
        )
        
        # Review and iterate on design
        reviewed_design = await self._review_trial_design(trial_design)
        
        # Save trial protocol
        await self.library.save_document(
            project_name,
            f"trial_protocol_v{self.iteration_count}",
            reviewed_design
        )
        
        await self.logger.log_phase_complete(
            project_name,
            "CLINICAL_TRIAL_DESIGN",
            {
                'trial_type': reviewed_design.get('study_type'),
                'duration': reviewed_design.get('duration'),
                'sample_size': reviewed_design.get('sample_size')
            }
        )
        
        return {
            'status': 'completed',
            'trial_design': reviewed_design,
            'next_phase': ResearchPhase.INNOVATION_PROPOSAL
        }
        
    async def _phase_innovation_proposal(self) -> Dict[str, Any]:
        """Phase 6: Generate innovative solutions and proposals."""
        project_name = self.current_project['name']
        
        await self.logger.log_phase_start(project_name, "INNOVATION_PROPOSAL", {
            'iteration': self.iteration_count
        })
        
        # Generate innovative ideas based on all findings
        innovations = await self._generate_innovations(
            self.hypotheses,
            self.findings,
            self.experiments
        )
        
        # Evaluate and prioritize innovations
        prioritized = await self._prioritize_innovations(innovations)
        
        # Create detailed proposal for top innovation
        if prioritized:
            proposal = await self._create_innovation_proposal(prioritized[0])
            
            self.innovations.append(proposal)
            await self.library.save_document(
                project_name,
                f"innovation_proposal_{len(self.innovations)}",
                proposal
            )
            
        await self.logger.log_phase_complete(
            project_name,
            "INNOVATION_PROPOSAL",
            {
                'innovations_generated': len(innovations),
                'top_innovation': prioritized[0]['title'] if prioritized else None
            }
        )
        
        return {
            'status': 'completed',
            'innovations': prioritized,
            'next_phase': ResearchPhase.LOGGING_PLANNING
        }
        
    async def _phase_logging_planning(self) -> Dict[str, Any]:
        """Phase 7: Log progress and plan next iteration."""
        project_name = self.current_project['name']
        
        await self.logger.log_phase_start(project_name, "LOGGING_PLANNING", {
            'iteration': self.iteration_count
        })
        
        # Create comprehensive progress report
        progress_report = await self._create_progress_report()
        
        # Determine if another iteration is needed
        should_continue = self._assess_research_completion()
        
        # Plan next iteration if continuing
        if should_continue:
            next_steps = await self._plan_next_iteration()
            self.iteration_count += 1
        else:
            next_steps = {'action': 'complete_research'}
            
        # Save iteration summary
        await self.library.save_document(
            project_name,
            f"iteration_{self.iteration_count}_summary",
            {
                'progress_report': progress_report,
                'next_steps': next_steps,
                'should_continue': should_continue
            }
        )
        
        await self.logger.log_phase_complete(
            project_name,
            "LOGGING_PLANNING",
            {
                'continue': should_continue,
                'next_action': next_steps.get('action')
            }
        )
        
        # Determine next phase
        if should_continue:
            next_phase = ResearchPhase.PROBLEM_DEFINITION  # Start new iteration
        else:
            next_phase = None  # Research complete
            
        return {
            'status': 'completed',
            'continue': should_continue,
            'next_phase': next_phase
        }
        
    async def _check_hourly_log(self):
        """Check if hourly log is needed and create it."""
        current_time = time.time()
        if current_time - self.last_log_time >= 3600:  # 1 hour
            await self._create_hourly_log()
            self.last_log_time = current_time
            
    async def _create_hourly_log(self):
        """Create detailed hourly progress log."""
        if not self.current_project:
            return
            
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'project': self.current_project['name'],
            'current_phase': self.current_phase.value if self.current_phase else None,
            'iteration': self.iteration_count,
            'recent_actions': await self._summarize_recent_actions(),
            'key_findings': await self._summarize_key_findings(),
            'current_hypothesis': self.hypotheses[0] if self.hypotheses else None,
            'thought_process': await self._explain_reasoning(),
            'next_steps': await self._outline_next_steps()
        }
        
        await self.logger.log_hourly_update(
            self.current_project['name'],
            log_entry
        )
        
    # Helper methods for each phase
    
    async def _search_local_knowledge(self, query: str) -> List[Dict]:
        """Search local knowledge base first."""
        # Use pharma RAG to search
        docs = self.pharma_rag.retrieve(query)
        
        # Convert to structured format
        structured_docs = []
        for i, doc in enumerate(docs[:10]):  # Limit to top 10
            structured_docs.append({
                'content': doc,
                'source': 'local',
                'relevance': 1.0 - (i * 0.1)  # Simple relevance scoring
            })
            
        return structured_docs
        
    async def _analyze_mechanisms(self, hypothesis: str, docs: List[Dict]) -> Dict:
        """Analyze biological mechanisms from literature."""
        # Combine document contents
        combined_docs = "\n\n".join([d['content'][:500] for d in docs[:5]])
        
        prompt = f"""
        Analyze the biological mechanisms related to this hypothesis:
        {hypothesis}
        
        Based on these research findings:
        {combined_docs}
        
        Provide:
        1. Key molecular/cellular mechanisms
        2. Physiological pathways involved
        3. Cause-effect relationships
        4. Supporting evidence strength
        
        Focus on sports performance and muscle biology.
        """
        
        response = await self.chat.generate(prompt)
        
        return {
            'analysis': response,
            'num_sources': len(docs),
            'timestamp': datetime.now().isoformat()
        }
        
    async def _search_external_sources(self, query: str, gaps: List[str]) -> List[Dict]:
        """Search external sources for missing information."""
        # This would integrate with external APIs
        # For now, return empty as placeholder
        logger.info(f"Would search external sources for: {query}")
        logger.info(f"To fill gaps: {gaps}")
        
        return []
        
    async def _synthesize_literature(self, hypothesis: str, docs: List[Dict], mechanisms: Dict) -> str:
        """Synthesize findings from literature review."""
        prompt = f"""
        Synthesize the literature findings for this hypothesis:
        {hypothesis}
        
        Mechanism analysis:
        {mechanisms.get('analysis', 'No mechanism analysis available')}
        
        Number of sources reviewed: {len(docs)}
        
        Provide a comprehensive synthesis that:
        1. Summarizes key supporting evidence
        2. Notes any contradicting findings
        3. Identifies remaining unknowns
        4. Suggests strength of evidence
        """
        
        return await self.chat.generate(prompt)
        
    def _extract_hypotheses(self, response: str) -> List[str]:
        """Extract hypotheses from LLM response."""
        # Simple extraction - look for numbered items
        lines = response.split('\n')
        hypotheses = []
        
        for line in lines:
            # Look for patterns like "1.", "2.", "Hypothesis:"
            if any(marker in line for marker in ['1.', '2.', '3.', 'Hypothesis:', 'H1:', 'H2:']):
                # Clean and add
                cleaned = line.split('.', 1)[-1].strip()
                if len(cleaned) > 20:  # Minimum length
                    hypotheses.append(cleaned)
                    
        return hypotheses[:3]  # Max 3 hypotheses
        
    async def _evaluate_hypothesis(self, hypothesis: str, findings: Dict) -> Dict:
        """Evaluate hypothesis against findings."""
        prompt = f"""
        Evaluate this hypothesis against the research findings:
        
        Hypothesis: {hypothesis}
        
        Key findings:
        {findings.get('synthesis', 'No synthesis available')}
        
        Coverage score: {findings.get('knowledge_gaps', {}).get('coverage_score', 0)}
        
        Provide:
        1. Support level (0-1)
        2. Key supporting evidence
        3. Key contradicting evidence
        4. Overall assessment
        """
        
        response = await self.chat.generate(prompt)
        
        # Extract support level (simplified)
        support_level = 0.5  # Default
        if 'strong support' in response.lower():
            support_level = 0.8
        elif 'weak support' in response.lower():
            support_level = 0.3
        elif 'no support' in response.lower():
            support_level = 0.1
            
        return {
            'support_level': support_level,
            'reason': response,
            'timestamp': datetime.now().isoformat()
        }
        
    async def _design_experiment(self, hypothesis: str) -> Dict:
        """Design an experiment to test the hypothesis."""
        prompt = f"""
        Design an experiment to test this hypothesis:
        {hypothesis}
        
        Create a detailed experimental design including:
        1. Objective and specific aims
        2. Methods (participants, interventions, measurements)
        3. Variables (independent, dependent, controls)
        4. Duration and timeline
        5. Expected outcomes
        6. Can this be simulated? (yes/no and how)
        
        Focus on practical sports/fitness experiments.
        """
        
        response = await self.chat.generate(prompt)
        
        # Determine if simulatable
        simulatable = 'simulat' in response.lower() and 'yes' in response.lower()
        
        return {
            'design': response,
            'simulatable': simulatable,
            'hypothesis': hypothesis,
            'timestamp': datetime.now().isoformat()
        }
        
    async def _run_simulation(self, experiment_design: Dict) -> Dict:
        """Run simulation based on experiment design."""
        # Extract parameters from design
        # This is simplified - real implementation would parse design
        
        simulation_type = "training_response"  # Default
        if 'supplement' in experiment_design.get('design', '').lower():
            simulation_type = "supplement_effect"
            
        # Run appropriate simulation
        results = await self.simulation_engine.run_simulation(
            simulation_type,
            {
                'duration_weeks': 8,
                'intervention': experiment_design.get('hypothesis', ''),
                'baseline': {'performance': 100}
            }
        )
        
        return results
        
    async def _analyze_experiment_results(self, experiment_design: Dict, simulation_results: Optional[Dict]) -> Dict[str, Any]:
        """Analyze experiment results."""
        if simulation_results and not simulation_results.get('error'):
            # Use simulation engine's analysis
            return await self.simulation_engine.analyze_experiment_results(
                experiment_design,
                simulation_results
            )
        else:
            # No simulation results - manual analysis
            return {
                'hypothesis_supported': False,
                'confidence': 0.3,
                'outcome': 'no_simulation',
                'recommendation': 'Consider manual experiment or revise design'
            }
        
    async def _design_clinical_trial(self, hypothesis: str, experiment: Dict = None) -> Dict:
        """Design a clinical trial protocol."""
        context = ""
        if experiment:
            context = f"""
            Previous experiment results:
            {experiment.get('analysis', {}).get('outcome', 'No results')}
            """
            
        prompt = f"""
        Design a clinical trial to test this hypothesis in humans:
        {hypothesis}
        
        {context}
        
        Include:
        1. Study objectives and hypotheses
        2. Study design (RCT, crossover, etc.)
        3. Population (inclusion/exclusion criteria)
        4. Sample size justification
        5. Intervention protocol
        6. Primary and secondary outcomes
        7. Duration and follow-up
        8. Safety monitoring plan
        9. Statistical analysis plan
        10. Ethical considerations
        
        Focus on sports performance or muscle/training studies.
        """
        
        response = await self.chat.generate(prompt)
        
        return {
            'protocol': response,
            'hypothesis': hypothesis,
            'study_type': 'RCT',  # Would be extracted from response
            'duration': '12 weeks',  # Would be extracted
            'sample_size': 40,  # Would be calculated
            'timestamp': datetime.now().isoformat()
        }
        
    async def _review_trial_design(self, design: Dict) -> Dict:
        """Review and improve trial design."""
        prompt = f"""
        Review this clinical trial design for quality and completeness:
        
        {design.get('protocol', '')}
        
        Check for:
        1. Scientific rigor
        2. Ethical considerations
        3. Feasibility
        4. Safety measures
        5. Statistical power
        
        Suggest improvements if needed.
        """
        
        review = await self.chat.generate(prompt)
        
        design['review'] = review
        design['version'] = 2
        
        return design
        
    async def _generate_innovations(self, hypotheses: List[str], findings: List[Dict], experiments: List[Dict]) -> List[Dict]:
        """Generate innovative solutions based on research."""
        context = f"""
        Research context:
        - Hypotheses tested: {len(hypotheses)}
        - Key findings: {len(findings)}
        - Experiments: {len(experiments)}
        
        Top hypothesis: {hypotheses[0] if hypotheses else 'None'}
        """
        
        prompt = f"""
        Based on this research, generate innovative solutions:
        
        {context}
        
        Create 3-5 innovative ideas for:
        1. New supplements or compounds
        2. Novel training protocols
        3. Performance enhancement strategies
        4. Recovery methods
        5. Practical applications
        
        Be creative but scientifically grounded.
        Focus on sports, fitness, and muscle performance.
        """
        
        response = await self.chat.generate(prompt)
        
        # Parse innovations (simplified)
        innovations = []
        ideas = response.split('\n\n')
        for i, idea in enumerate(ideas[:5]):
            if len(idea) > 50:
                innovations.append({
                    'id': i + 1,
                    'description': idea,
                    'category': 'performance',
                    'novelty': 0.7,  # Would be assessed
                    'feasibility': 0.8  # Would be assessed
                })
                
        return innovations
        
    async def _prioritize_innovations(self, innovations: List[Dict]) -> List[Dict]:
        """Prioritize innovations based on impact and feasibility."""
        # Simple scoring
        for innovation in innovations:
            innovation['score'] = (
                innovation.get('novelty', 0.5) * 0.4 +
                innovation.get('feasibility', 0.5) * 0.6
            )
            
        # Sort by score
        return sorted(innovations, key=lambda x: x['score'], reverse=True)
        
    async def _create_innovation_proposal(self, innovation: Dict) -> Dict:
        """Create detailed proposal for top innovation."""
        prompt = f"""
        Create a detailed implementation proposal for this innovation:
        
        {innovation.get('description', '')}
        
        Include:
        1. Executive summary
        2. Scientific rationale
        3. Implementation plan
        4. Required resources
        5. Timeline
        6. Expected outcomes
        7. Risk assessment
        8. Next steps
        """
        
        proposal = await self.chat.generate(prompt)
        
        return {
            'innovation': innovation,
            'proposal': proposal,
            'title': f"Innovation Proposal {innovation.get('id', 1)}",
            'timestamp': datetime.now().isoformat()
        }
        
    async def _create_progress_report(self) -> Dict:
        """Create comprehensive progress report."""
        return {
            'project': self.current_project['name'],
            'iteration': self.iteration_count,
            'phases_completed': len(self.current_project.get('phases_completed', [])),
            'hypotheses_tested': len(self.hypotheses),
            'key_findings': len(self.findings),
            'experiments_run': len(self.experiments),
            'innovations_proposed': len(self.innovations),
            'current_status': self.status.value,
            'timestamp': datetime.now().isoformat()
        }
        
    def _assess_research_completion(self) -> bool:
        """Assess if research should continue or complete."""
        # Continue if:
        # - Less than 5 iterations
        # - Still have untested hypotheses
        # - Recent findings suggest new directions
        
        if self.iteration_count >= 5:
            return False
            
        if len(self.hypotheses) > len(self.experiments):
            return True
            
        if self.findings and self.findings[-1].get('knowledge_gaps', {}).get('coverage_score', 1) < 0.8:
            return True
            
        return False
        
    async def _plan_next_iteration(self) -> Dict:
        """Plan the next research iteration."""
        # Determine focus for next iteration
        if self.findings and self.findings[-1].get('knowledge_gaps'):
            gaps = self.findings[-1]['knowledge_gaps']['missing_aspects']
            focus = f"Fill knowledge gaps: {', '.join(gaps[:2])}"
        else:
            focus = "Explore alternative hypotheses"
            
        return {
            'action': 'continue_research',
            'focus': focus,
            'planned_phases': ['PROBLEM_DEFINITION', 'LITERATURE_REVIEW'],
            'estimated_time': '2-3 hours'
        }
        
    def _determine_next_phase(self, phase_result: Dict) -> Optional[ResearchPhase]:
        """Determine the next research phase based on results."""
        return phase_result.get('next_phase')
        
    def _should_complete_research(self) -> bool:
        """Check if research should be completed."""
        return self.current_phase is None or self.iteration_count > 10
        
    async def _complete_research(self):
        """Complete the research project."""
        self.status = ResearchStatus.COMPLETED
        
        # Create final report
        final_report = {
            'project': self.current_project,
            'total_iterations': self.iteration_count,
            'hypotheses': self.hypotheses,
            'key_findings': self.findings,
            'experiments': self.experiments,
            'innovations': self.innovations,
            'completion_time': datetime.now().isoformat()
        }
        
        await self.library.save_document(
            self.current_project['name'],
            'final_report',
            final_report
        )
        
        await self.logger.log_event(
            self.current_project['name'],
            'PROJECT_COMPLETE',
            {
                'total_duration': str(datetime.now() - self.current_project['start_time']),
                'outcomes': len(self.innovations)
            }
        )
        
    # Methods for hourly logging
    
    async def _summarize_recent_actions(self) -> str:
        """Summarize recent research actions."""
        recent_logs = await self.logger.get_recent_logs(
            self.current_project['name'],
            hours=1
        )
        
        actions = []
        for log in recent_logs:
            if log.get('event_type') == 'PHASE_COMPLETE':
                actions.append(f"Completed {log.get('phase', 'unknown')} phase")
                
        return "; ".join(actions) if actions else "Continuing current phase"
        
    async def _summarize_key_findings(self) -> List[str]:
        """Summarize key findings from recent work."""
        if not self.findings:
            return ["No findings yet"]
            
        recent = self.findings[-1]
        return [
            f"Coverage score: {recent.get('knowledge_gaps', {}).get('coverage_score', 0):.1%}",
            f"Sources reviewed: {recent.get('local_sources', 0) + recent.get('external_sources', 0)}"
        ]
        
    async def _explain_reasoning(self) -> str:
        """Explain current reasoning and decision-making."""
        if self.current_phase == ResearchPhase.LITERATURE_REVIEW:
            return "Searching literature to understand mechanisms and prior work"
        elif self.current_phase == ResearchPhase.EXPERIMENT_DESIGN:
            return "Designing experiments to test hypothesis with simulations"
        else:
            return f"Executing {self.current_phase.value} to advance research"
            
    async def _outline_next_steps(self) -> str:
        """Outline planned next steps."""
        if self.current_phase:
            next_phase = self._determine_next_phase({'next_phase': self.current_phase})
            return f"Next: {next_phase.value if next_phase else 'Complete research'}"
        return "Determining next research direction"
        
    # Public methods for monitoring
    
    async def get_current_status(self) -> Dict[str, Any]:
        """Get current research status for monitoring."""
        return {
            'project': self.current_project,
            'status': self.status.value,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'iteration': self.iteration_count,
            'hypotheses_count': len(self.hypotheses),
            'findings_count': len(self.findings),
            'last_update': datetime.now().isoformat()
        }
        
    async def pause_research(self):
        """Pause the research."""
        self.status = ResearchStatus.PAUSED
        await self.logger.log_event(
            self.current_project['name'],
            'RESEARCH_PAUSED',
            {'reason': 'User requested'}
        )
        
    async def resume_research(self):
        """Resume paused research."""
        if self.status == ResearchStatus.PAUSED:
            self.status = ResearchStatus.ACTIVE
            await self.logger.log_event(
                self.current_project['name'],
                'RESEARCH_RESUMED',
                {}
            )
            # Restart research loop
            asyncio.create_task(self._run_research_loop())

    async def _generate_alternative_hypothesis(self, original: str, findings: Dict) -> str:
        """Generate an alternative hypothesis based on findings."""
        prompt = f"""
        The original hypothesis was not well supported:
        {original}
        
        Based on these findings:
        {findings.get('synthesis', 'No synthesis available')}
        
        Generate an alternative hypothesis that better fits the evidence.
        Focus on sports performance and muscle biology.
        """
        
        return await self.chat.generate(prompt)
        
    async def _refine_hypothesis(self, hypothesis: str, findings: Dict) -> str:
        """Refine the hypothesis based on findings."""
        prompt = f"""
        Refine this hypothesis based on new evidence:
        {hypothesis}
        
        Evidence summary:
        {findings.get('synthesis', 'No synthesis available')}
        
        Provide a more specific or nuanced version of the hypothesis.
        """
        
        return await self.chat.generate(prompt) 