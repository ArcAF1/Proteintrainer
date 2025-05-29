"""Build Neo4j graph from ingested data."""
from __future__ import annotations

from typing import Iterable
from neo4j import Driver

from .neo4j_setup import get_driver


def build_graph(articles: Iterable[dict], relations: Iterable[dict]) -> None:
    """Insert nodes and relationships into Neo4j."""
    driver: Driver = get_driver()
    with driver.session() as session:
        for art in articles:
            session.run(
                "MERGE (a:Article {id:$id, title:$title})",
                id=art["id"],
                title=art.get("title", ""),
            )
            for ent in art.get("entities", []):
                label = ent.get("type", "Concept")
                session.run(
                    f"MERGE (e:{label} {{name:$name}})",
                    name=ent["name"],
                )
                session.run(
                    f"MATCH (a:Article {{id:$id}}), (e:{label} {{name:$name}}) MERGE (a)-[:MENTIONS]->(e)",
                    id=art["id"],
                    name=ent["name"],
                )
        for rel in relations:
            label1 = rel.get("type1", "Concept")
            label2 = rel.get("type2", "Concept")
            session.run(
                f"MERGE (s:{label1} {{name:$s}})",
                s=rel["source"],
            )
            session.run(
                f"MERGE (t:{label2} {{name:$t}})",
                t=rel["target"],
            )
            session.run(
                f"MATCH (s:{label1} {{name:$s}}), (t:{label2} {{name:$t}}) "
                f"MERGE (s)-[r:{rel['rel'].upper()}]->(t)" +
                (" SET r.article_id=$aid" if rel.get("article_id") else ""),
                s=rel["source"],
                t=rel["target"],
                aid=rel.get("article_id"),
            )
    driver.close()


if __name__ == "__main__":
    sample_articles = [
        {"id": "A1", "title": "Creatine Study", "entities": [{"name": "creatine", "type": "Drug"}, {"name": "insulin", "type": "Concept"}]}
    ]
    sample_rels = [
        {"source": "creatine", "type1": "Drug", "rel": "AFFECTS", "target": "insulin", "type2": "Concept", "article_id": "A1"}
    ]
    build_graph(sample_articles, sample_rels)
