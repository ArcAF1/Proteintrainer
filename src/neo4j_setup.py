"""Neo4j setup and connection management."""

import os
from neo4j import GraphDatabase
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connection settings from environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_driver: Optional[GraphDatabase.driver] = None


def get_driver():
    """Get or create Neo4j driver instance."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Test connection
        with _driver.session() as session:
            session.run("RETURN 1")
    return _driver


def close_driver():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
