"""Neo4j connection utilities."""
from __future__ import annotations

import os
from neo4j import GraphDatabase, Driver


_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_USER = os.getenv("NEO4J_USER", "neo4j")
_PASS = os.getenv("NEO4J_PASS", "test")


def get_driver() -> Driver:
    """Return a Neo4j driver instance."""
    driver = GraphDatabase.driver(_URI, auth=(_USER, _PASS))
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS ON (n:Concept) ASSERT n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS ON (n:Drug) ASSERT n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS ON (n:Article) ASSERT n.id IS UNIQUE")
    return driver
