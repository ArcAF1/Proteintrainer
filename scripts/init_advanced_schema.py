#!/usr/bin/env python3
"""Initialize advanced schema for Neo4j biomedical knowledge graph.

Run after starting Neo4j container to create constraints, indexes, and node types.
"""
import os
import sys
from pathlib import Path
import yaml
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load schema from YAML
SCHEMA_FILE = Path(__file__).parent / 'init_advanced_schema.yaml'

def run_statements(driver, statements):
    """Run a list of Cypher statements."""
    with driver.session() as session:
        for stmt in statements:
            try:
                logger.info(f"Running: {stmt[:60]}...")
                session.run(stmt)
                logger.info("✓ Success")
            except Exception as e:
                logger.error(f"✗ Failed: {e}")
                # Continue with other statements

def main():
    # Get connection details from environment
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    
    logger.info(f"Connecting to Neo4j at {uri} as user '{user}'...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()
        logger.info("Successfully connected to Neo4j")
        
        # Load and apply schema
        if SCHEMA_FILE.exists():
            with open(SCHEMA_FILE) as f:
                schema = yaml.safe_load(f)
            
            logger.info("Creating constraints...")
            run_statements(driver, schema.get('constraints', []))
            
            logger.info("Creating indexes...")
            run_statements(driver, schema.get('indexes', []))
            
            logger.info("Creating fulltext indexes...")
            run_statements(driver, schema.get('fulltext_indexes', []))
        else:
            logger.warning(f"Schema file not found: {SCHEMA_FILE}")
        
        driver.close()
        logger.info("Schema initialization complete!")
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.error("Make sure Neo4j is running and credentials are correct in .env file")
        sys.exit(1)

if __name__ == "__main__":
    main() 