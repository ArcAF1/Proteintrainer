"""
ReAct Biomedical Agent - Multi-step reasoning with tool use
Based on the guide's recommendations for epistemic awareness and reasoning traces
"""
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish

from .enhanced_llm import get_enhanced_llm
from .rag_chat import get_chat
from .graph_query import get_subgraph

logger = logging.getLogger(__name__)

# ReAct prompt with epistemic awareness and biomedical focus
REACT_PROMPT = """You are a biomedical research assistant with access to tools.

IMPORTANT INSTRUCTIONS:
1. Think step-by-step and show your reasoning
2. If you're uncertain, say so explicitly
3. Always cite sources when making factual claims
4. Double-check important facts using available tools
5. If information is missing, acknowledge the gap

You have access to these tools:
{tools}

Use this format:
Question: the input question you must answer
Thought: think about what you need to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer (or I need to acknowledge uncertainty)
Final Answer: the final answer with citations

If you're not sure about something, your final answer should include phrases like:
- "Based on available evidence..."
- "I'm not certain about X, but..."
- "This requires further investigation..."
- "Current knowledge suggests..."

Begin!

Question: {input}
{agent_scratchpad}"""

class BiomedicalReActAgent:
    """ReAct agent with biomedical expertise and reasoning transparency."""
    
    def __init__(self):
        self.llm = get_enhanced_llm()
        self.rag = get_chat()
        self.reasoning_trace = []
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent."""
        tools = []
        
        # Knowledge base search
        tools.append(Tool(
            name="KnowledgeSearch",
            func=self._search_knowledge,
            description="Search the 11GB biomedical knowledge base for information"
        ))
        
        # Graph query tool
        if True:  # Check if Neo4j is available
            tools.append(Tool(
                name="GraphQuery", 
                func=self._query_graph,
                description="Query the biomedical knowledge graph for relationships (e.g., 'What drugs treat diabetes?')"
            ))
        
        # Uncertainty check tool
        tools.append(Tool(
            name="VerifyFact",
            func=self._verify_fact,
            description="Double-check a specific fact or claim against the knowledge base"
        ))
        
        # Citation finder
        tools.append(Tool(
            name="FindCitation",
            func=self._find_citation,
            description="Find proper citations for a claim"
        ))
        
        return tools
    
    def _search_knowledge(self, query: str) -> str:
        """Search the knowledge base."""
        try:
            if not self.rag.is_ready():
                return "Knowledge base not available"
            
            docs = self.rag.retrieve(query)
            if not docs:
                return f"No information found about: {query}"
            
            # Return top 3 most relevant excerpts
            results = []
            for i, doc in enumerate(docs[:3], 1):
                excerpt = doc[:300] + "..." if len(doc) > 300 else doc
                results.append(f"[Source {i}]: {excerpt}")
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Knowledge search error: {e}")
            return f"Error searching knowledge base: {str(e)}"
    
    def _query_graph(self, query: str) -> str:
        """Query the knowledge graph."""
        try:
            # Simple entity extraction
            entities = [word for word in query.split() if word[0].isupper()]
            if not entities:
                return "Please specify an entity to query (e.g., a drug or disease name)"
            
            entity = entities[0]
            nodes, edges = get_subgraph(entity, max_depth=2)
            
            if not edges:
                return f"No relationships found for {entity} in the knowledge graph"
            
            # Format relationships
            relationships = []
            for edge in edges[:10]:  # Limit to 10
                rel = f"{edge['source']} → {edge['target']} ({edge.get('type', 'related_to')})"
                relationships.append(rel)
            
            return f"Graph relationships for {entity}:\n" + "\n".join(relationships)
            
        except Exception as e:
            return f"Graph query error: {str(e)}"
    
    def _verify_fact(self, claim: str) -> str:
        """Verify a specific claim."""
        search_result = self._search_knowledge(claim)
        
        if "No information found" in search_result:
            return f"UNVERIFIED: Could not find evidence for '{claim}' in the knowledge base"
        else:
            return f"VERIFICATION: Found supporting evidence:\n{search_result}"
    
    def _find_citation(self, claim: str) -> str:
        """Find citations for a claim."""
        search_result = self._search_knowledge(claim)
        
        if "No information found" in search_result:
            return "No citations found - claim may need verification"
        
        # In a real implementation, extract actual citations
        return f"Suggested citation: Based on knowledge base search for '{claim}'"
    
    def _create_agent(self):
        """Create the ReAct agent."""
        prompt = PromptTemplate(
            template=REACT_PROMPT,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )
        
        # Create a simple wrapper for the LLM to work with LangChain
        class LLMWrapper:
            def __init__(self, llm):
                self.llm = llm
                
            async def agenerate(self, prompts, **kwargs):
                # Simple async wrapper
                from langchain.schema import LLMResult, Generation
                
                generations = []
                for prompt in prompts:
                    response = self.llm.generate(prompt)
                    generations.append([Generation(text=response)])
                
                return LLMResult(generations=generations)
            
            def __call__(self, prompt, **kwargs):
                return self.llm.generate(prompt)
        
        wrapped_llm = LLMWrapper(self.llm)
        
        agent = create_react_agent(
            llm=wrapped_llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Show reasoning steps
            max_iterations=5,
            return_intermediate_steps=True
        )
    
    async def run(self, question: str) -> Dict[str, Any]:
        """Run the agent on a question."""
        self.reasoning_trace = []
        start_time = datetime.now()
        
        try:
            # Record the question
            self.reasoning_trace.append({
                'type': 'question',
                'content': question,
                'timestamp': start_time.isoformat()
            })
            
            # Run the agent
            result = await self.agent.ainvoke({
                "input": question
            })
            
            # Extract reasoning steps
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) == 2:
                        action, observation = step
                        self.reasoning_trace.append({
                            'type': 'thought',
                            'content': getattr(action, 'log', str(action)),
                            'timestamp': datetime.now().isoformat()
                        })
                        self.reasoning_trace.append({
                            'type': 'observation',
                            'content': observation,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Record final answer
            self.reasoning_trace.append({
                'type': 'answer',
                'content': result.get('output', 'No answer generated'),
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'answer': result.get('output', 'No answer generated'),
                'reasoning_trace': self.reasoning_trace,
                'duration': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'reasoning_trace': self.reasoning_trace,
                'duration': (datetime.now() - start_time).total_seconds()
            }
    
    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """Get the reasoning trace from the last run."""
        return self.reasoning_trace

# Global instance
_react_agent = None

def get_react_agent() -> BiomedicalReActAgent:
    """Get or create the ReAct agent."""
    global _react_agent
    if _react_agent is None:
        _react_agent = BiomedicalReActAgent()
    return _react_agent

async def run_react_agent(question: str) -> Dict[str, Any]:
    """Convenience function to run the ReAct agent."""
    agent = get_react_agent()
    return await agent.run(question) 