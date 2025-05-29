"""Query functions for Neo4j graph."""
from __future__ import annotations

from typing import List

from neo4j import Driver

from .neo4j_setup import get_driver


def search_entity(name: str) -> List[str]:
    driver: Driver = get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE toLower(n.name) = toLower($name) RETURN labels(n)[0] AS label, n.name AS name",
            name=name,
        )
        record = result.single()
        if not record:
            return []
        label = record["label"]
        output = []
        rels = session.run(
            "MATCH (n {name:$name})-[r]-(m) RETURN type(r) AS rel, labels(m)[0] AS mlabel, m.name AS mname",
            name=record["name"],
        )
        for row in rels:
            output.append(f"({label}) {name} -[{row['rel']}]-> ({row['mlabel']}) {row['mname']}")
    driver.close()
    return output


def find_connections(name1: str, name2: str) -> str:
    driver: Driver = get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH p=shortestPath((a {name:$a})-[*..4]-(b {name:$b})) RETURN p",
            a=name1,
            b=name2,
        )
        record = result.single()
        if not record:
            return f"No connection found between {name1} and {name2}."
        nodes = [n["name"] for n in record["p"].nodes]
        rels = [r.type for r in record["p"].relationships]
    driver.close()
    path_str = nodes[0]
    for r, n in zip(rels, nodes[1:]):
        path_str += f" --{r}--> {n}"
    return path_str


if __name__ == "__main__":
    print(search_entity("creatine"))
    print(find_connections("creatine", "insulin"))
