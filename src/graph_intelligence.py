from __future__ import annotations
"""High-level graph query helpers for biomedical analysis."""
from typing import List
from neo4j import Driver
from .neo4j_setup import get_driver

class BiomedicalGraphQuerier:
    def __init__(self, driver: Driver | None = None):
        self.driver = driver or get_driver()

    def find_drug_pathways(self, drug: str, max_hops: int = 3) -> List[dict]:
        cypher = (
            "MATCH path = (d:Drug {name:$drug})-[*1.." + str(max_hops) + "]-(p:Pathway) "
            "WHERE ALL(r IN relationships(path) WHERE r.confidence > 0.7) "
            "RETURN path AS path, reduce(conf = 1.0, r IN relationships(path) | conf * r.confidence) AS conf "
            "ORDER BY conf DESC LIMIT 10"
        )
        with self.driver.session() as session:
            res = session.run(cypher, drug=drug)
            return [
                {
                    "nodes": [n["name"] or n.id for n in rec["path"].nodes],
                    "conf": rec["conf"],
                }
                for rec in res
            ]

    def close(self):
        self.driver.close() 