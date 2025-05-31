from __future__ import annotations
"""Simple LangChain agent wrapping PubMed / arXiv search tools.

Not used by GUI yet, but available for programmatic calls or future expansion.
"""
from typing import List

try:
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain_community.llms import CTransformers
    from langchain_community.utilities import SerpAPIWrapper  # Updated import
except Exception:  # pragma: no cover
    initialize_agent = None  # type: ignore

import requests
from .config import settings


PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ARXIV_API = "http://export.arxiv.org/api/query"

def pubmed_search(query: str) -> str:
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": "3"}
    try:
        r = requests.get(PUBMED_SEARCH, params=params, timeout=20)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        return ", ".join(ids[:3]) or "No result"
    except Exception as exc:  # pragma: no cover
        return f"Error: {exc}"

def arxiv_search(query: str) -> str:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": 3}
    try:
        r = requests.get(ARXIV_API, params=params, timeout=20)
        r.raise_for_status()
        import feedparser  # lazy import
        entries = feedparser.parse(r.text).entries
        return "; ".join(e.title for e in entries)
    except Exception as exc:  # pragma: no cover
        return f"Error: {exc}"


if initialize_agent is not None:
    llm = CTransformers(
        model=str(settings.model_dir / settings.llm_model),
        model_type="llama",
        max_new_tokens=256,
    )
    tools = [
        Tool(name="PubMedSearch", func=pubmed_search, description="search PubMed ids"),
        Tool(name="ArxivSearch", func=arxiv_search, description="search arXiv titles"),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
else:
    agent = None


def ask_agent(question: str) -> str:
    if agent is None:
        return "Agent unavailable"
    return agent.run(question) 