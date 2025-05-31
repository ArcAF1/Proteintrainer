"""Query functions for Neo4j graph."""
from __future__ import annotations

from typing import List, Tuple, Dict

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


def get_subgraph(entity_name: str, max_depth: int = 2) -> Tuple[List[Dict], List[Dict]]:
    """Get subgraph around an entity for visualization.
    
    Returns:
        Tuple of (nodes, edges) where:
        - nodes: List of dicts with 'name' and 'type' keys
        - edges: List of dicts with 'source', 'target', and 'type' keys
    """
    driver: Driver = get_driver()
    nodes = []
    edges = []
    node_set = set()
    
    try:
        with driver.session() as session:
            # Get nodes and relationships within max_depth of the entity
            result = session.run(
                f"""
                MATCH (n {{name: $name}})
                CALL apoc.path.subgraphAll(n, {{
                    maxLevel: $depth,
                    relationshipFilter: null,
                    labelFilter: null
                }})
                YIELD nodes, relationships
                RETURN nodes, relationships
                """,
                name=entity_name,
                depth=max_depth
            )
            
            record = result.single()
            if not record:
                # Fallback to simpler query if APOC is not available
                result = session.run(
                    """
                    MATCH (n {name: $name})
                    OPTIONAL MATCH (n)-[r]-(m)
                    RETURN n, collect(DISTINCT m) as connected_nodes, collect(DISTINCT r) as relationships
                    """,
                    name=entity_name
                )
                record = result.single()
                if record:
                    # Add central node
                    central_node = record['n']
                    node_name = central_node.get('name', str(central_node.id))
                    node_type = list(central_node.labels)[0] if central_node.labels else 'Unknown'
                    nodes.append({'name': node_name, 'type': node_type})
                    node_set.add(node_name)
                    
                    # Add connected nodes
                    for node in record['connected_nodes']:
                        if node:
                            node_name = node.get('name', str(node.id))
                            node_type = list(node.labels)[0] if node.labels else 'Unknown'
                            if node_name not in node_set:
                                nodes.append({'name': node_name, 'type': node_type})
                                node_set.add(node_name)
                    
                    # Add relationships
                    for rel in record['relationships']:
                        if rel:
                            start_node = rel.start_node
                            end_node = rel.end_node
                            source = start_node.get('name', str(start_node.id))
                            target = end_node.get('name', str(end_node.id))
                            edges.append({
                                'source': source,
                                'target': target,
                                'type': rel.type
                            })
            else:
                # Process APOC results
                for node in record['nodes']:
                    node_name = node.get('name', str(node.id))
                    node_type = list(node.labels)[0] if node.labels else 'Unknown'
                    if node_name not in node_set:
                        nodes.append({'name': node_name, 'type': node_type})
                        node_set.add(node_name)
                
                for rel in record['relationships']:
                    start_node = rel.start_node
                    end_node = rel.end_node
                    source = start_node.get('name', str(start_node.id))
                    target = end_node.get('name', str(end_node.id))
                    edges.append({
                        'source': source,
                        'target': target,
                        'type': rel.type
                    })
    
    except Exception as e:
        print(f"Error getting subgraph: {e}")
        # Return minimal data if there's an error
        nodes = [{'name': entity_name, 'type': 'Unknown'}]
        edges = []
    
    finally:
        driver.close()
    
    return nodes, edges


if __name__ == "__main__":
    print(search_entity("creatine"))
    print(find_connections("creatine", "insulin"))
    nodes, edges = get_subgraph("creatine")
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
