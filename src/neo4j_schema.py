from __future__ import annotations
"""Schema deployment for Neo4j biomedical graph.

Run once at startup or after Neo4j container init:
    python -m src.neo4j_schema
"""
import structlog
from neo4j import Driver
from .neo4j_setup import get_driver

log = structlog.get_logger(__name__)

NODE_LABELS = [
    "Drug", "Disease", "Gene", "Protein", "Pathway", "CellType", "Tissue",
    "SideEffect", "Symptom", "Compound", "Study", "Article", "Variant", "Organ",
    "Process",
]

REL_TYPES = [
    "TREATS", "CAUSES", "INTERACTS_WITH", "EXPRESSED_IN", "PART_OF", "IS_A",
    "ASSOCIATED_WITH", "INHIBITS", "ACTIVATES", "BINDS", "UPREGULATES", "DOWNREGULATES",
    "COEXPRESSED", "CORRELATED_WITH", "INVOLVED_IN", "TARGETS", "ALTERNATIVE_TO",
    "OBSERVED_IN", "LOCATED_IN", "RELATED_TO",
]

INDEX_LABELS = [
    ("Drug", "name"),
    ("Disease", "name"),
    ("Gene", "symbol"),
    ("Protein", "uniprot_id"),
]

PROPERTY_CONSTRAINTS = [
    ("Entity", "id"),
]

def deploy_schema(driver: Driver | None = None) -> None:
    if driver is None:
        driver = get_driver()
    with driver.session() as session:
        for lbl, prop in INDEX_LABELS:
            cypher = (
                f"CREATE INDEX IF NOT EXISTS FOR (n:{lbl}) ON (n.{prop})"  # noqa: E501
            )
            session.run(cypher)
            log.info("index", label=lbl, prop=prop)
        for label, prop in PROPERTY_CONSTRAINTS:
            cypher = (
                f"CREATE CONSTRAINT IF NOT EXISTS ON (n:{label}) ASSERT n.{prop} IS UNIQUE"
            )
            session.run(cypher)
            log.info("constraint", label=label, prop=prop)
    log.info("schema-deploy-complete")

def main() -> None:  # pragma: no cover
    deploy_schema()


if __name__ == "__main__":
    main() 