from __future__ import annotations
"""Graph-aware RAG pipeline.

Uses LangChain `GraphCypherQAChain` to let the LLM generate Cypher queries for
Neo4j and combine those results with vector-look-ups.

The code is written to *degrade gracefully* in environments where LangChain or
Neo4j are not installed.  In that case, `GraphRAG.available` will be `False`
and callers should fall back to normal RAG.
"""
from typing import Any, List

from .config import settings

try:
    # LangChain ≥0.1.0 uses separate community packages for integrations.
    from langchain_community.graphs import Neo4jGraph
    from langchain.chains import GraphCypherQAChain
    from langchain_community.llms import CTransformers
except Exception:  # pragma: no cover
    Neo4jGraph = None  # type: ignore
    GraphCypherQAChain = None  # type: ignore
    CTransformers = None  # type: ignore


class GraphRAG:  # pylint: disable=too-few-public-methods
    """Wrapper around LangChain GraphCypherQAChain."""

    available: bool = GraphCypherQAChain is not None and Neo4jGraph is not None

    def __init__(self) -> None:
        if not self.available:
            raise RuntimeError("GraphRAG unavailable – missing LangChain / Neo4j libs")

        # Build Neo4jGraph interface (auto-connect to bolt URL)
        self.graph = Neo4jGraph(
            url=settings.neo4j_uri, 
            username=settings.neo4j_user, 
            password=settings.neo4j_password
        )

        # Create LLM wrapper around local GGUF model via ctransformers
        self.llm = CTransformers(
            model=str(settings.model_dir / settings.llm_model),
            model_type="llama",
            max_new_tokens=256,
        )

        self.chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            verbose=False,
            top_k=5,
        )

    def answer(self, question: str) -> str:  # noqa: D401 (simple)
        """Return a graph-augmented answer string."""
        return self.chain.run(question)


# ---------------------------------------------------------------------------
# Helper so GUI can hide tabs when GraphRAG isn't usable


def graphrag_available() -> bool:  # noqa: D401
    """Return True if GraphRAG can be instantiated in this environment."""
    return GraphRAG.available


__all__: List[str] = ["GraphRAG", "graphrag_available"] 