"""Neo4j connection utilities with retry logic."""

import time
import logging
from neo4j import GraphDatabase
from contextlib import contextmanager
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j connection with retry logic."""
    
    def __init__(self):
        self.driver: Optional[GraphDatabase.driver] = None
        
    def connect(self) -> bool:
        """Connect to Neo4j with retry logic."""
        for attempt in range(settings.neo4j_max_retries):
            try:
                logger.info(f"Attempting to connect to Neo4j (attempt {attempt + 1}/{settings.neo4j_max_retries})")
                
                self.driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                    connection_timeout=settings.neo4j_connection_timeout
                )
                
                # Test the connection
                with self.driver.session() as session:
                    result = session.run("RETURN 1 AS test")
                    result.single()
                
                logger.info("Successfully connected to Neo4j")
                return True
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < settings.neo4j_max_retries - 1:
                    logger.info(f"Retrying in {settings.neo4j_retry_delay} seconds...")
                    time.sleep(settings.neo4j_retry_delay)
                else:
                    logger.error(f"Failed to connect to Neo4j after {settings.neo4j_max_retries} attempts")
                    return False
        
        return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                session.run("RETURN 1").single()
            return True
        except:
            return False


# Global connection instance
neo4j_conn = Neo4jConnection()


def ensure_neo4j_connected():
    """Ensure Neo4j is connected, retry if not."""
    if not neo4j_conn.is_connected():
        logger.info("Neo4j connection lost, attempting to reconnect...")
        return neo4j_conn.connect()
    return True 