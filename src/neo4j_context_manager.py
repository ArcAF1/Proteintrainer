"""
Neo4j-Based Context Manager
Advanced memory system using Neo4j for conversation context,
entity relationships, and research history
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib

from neo4j import Driver
from .neo4j_setup import get_driver
from .settings import settings

logger = logging.getLogger(__name__)


class Neo4jContextManager:
    """
    Manages conversation context and research history in Neo4j.
    Tracks entities, relationships, hypotheses, and findings over time.
    """
    
    def __init__(self, driver: Optional[Driver] = None):
        """Initialize with Neo4j driver."""
        self.driver = driver or get_driver()
        self._ensure_constraints()
        
    def _ensure_constraints(self):
        """Create Neo4j constraints and indexes for optimal performance."""
        try:
            with self.driver.session() as session:
                # Constraints for uniqueness
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Finding) REQUIRE f.id IS UNIQUE"
                ]
                
                # Indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS FOR (m:Message) ON (m.timestamp)",
                    "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX IF NOT EXISTS FOR (h:Hypothesis) ON (h.status)",
                    "CREATE INDEX IF NOT EXISTS FOR (f:Finding) ON (f.confidence)"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.debug(f"Constraint already exists or error: {e}")
                        
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index already exists or error: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to create constraints/indexes: {e}")
            
    def create_conversation(self, user_id: str, topic: str) -> str:
        """Create a new conversation node."""
        conversation_id = hashlib.md5(
            f"{user_id}_{topic}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        with self.driver.session() as session:
            session.run("""
                CREATE (c:Conversation {
                    id: $id,
                    user_id: $user_id,
                    topic: $topic,
                    created_at: datetime(),
                    status: 'active'
                })
            """, id=conversation_id, user_id=user_id, topic=topic)
            
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id
        
    def add_message(self, 
                   conversation_id: str,
                   role: str,
                   content: str,
                   entities: Optional[List[str]] = None) -> str:
        """Add a message to the conversation and extract entities."""
        message_id = hashlib.md5(
            f"{conversation_id}_{role}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        with self.driver.session() as session:
            # Create message
            session.run("""
                MATCH (c:Conversation {id: $conv_id})
                CREATE (m:Message {
                    id: $msg_id,
                    role: $role,
                    content: $content,
                    timestamp: datetime()
                })
                CREATE (c)-[:HAS_MESSAGE]->(m)
            """, conv_id=conversation_id, msg_id=message_id, 
                role=role, content=content)
            
            # Link to entities if provided
            if entities:
                for entity in entities:
                    self._link_entity_to_message(session, message_id, entity)
                    
        return message_id
        
    def _link_entity_to_message(self, session, message_id: str, entity_name: str):
        """Link an entity to a message, creating if necessary."""
        # Determine entity type
        entity_type = self._classify_entity(entity_name)
        
        session.run("""
            MATCH (m:Message {id: $msg_id})
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.type = $type, e.created_at = datetime()
            CREATE (m)-[:MENTIONS]->(e)
        """, msg_id=message_id, name=entity_name, type=entity_type)
        
    def _classify_entity(self, entity_name: str) -> str:
        """Simple entity classification."""
        entity_lower = entity_name.lower()
        
        if any(term in entity_lower for term in ['protein', 'enzyme', 'receptor']):
            return 'protein'
        elif any(term in entity_lower for term in ['muscle', 'tissue', 'organ']):
            return 'anatomy'
        elif any(term in entity_lower for term in ['drug', 'supplement', 'compound']):
            return 'compound'
        elif any(term in entity_lower for term in ['pathway', 'process', 'mechanism']):
            return 'biological_process'
        else:
            return 'general'
            
    def store_hypothesis(self,
                        conversation_id: str,
                        hypothesis: str,
                        confidence: float = 0.5,
                        entities: Optional[List[str]] = None) -> str:
        """Store a research hypothesis in the graph."""
        hypothesis_id = hashlib.md5(
            f"{hypothesis}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        with self.driver.session() as session:
            # Create hypothesis
            session.run("""
                MATCH (c:Conversation {id: $conv_id})
                CREATE (h:Hypothesis {
                    id: $hyp_id,
                    content: $content,
                    confidence: $confidence,
                    status: 'proposed',
                    created_at: datetime()
                })
                CREATE (c)-[:GENERATED]->(h)
            """, conv_id=conversation_id, hyp_id=hypothesis_id,
                content=hypothesis, confidence=confidence)
            
            # Link to entities
            if entities:
                for entity in entities:
                    session.run("""
                        MATCH (h:Hypothesis {id: $hyp_id})
                        MERGE (e:Entity {name: $name})
                        CREATE (h)-[:INVOLVES]->(e)
                    """, hyp_id=hypothesis_id, name=entity)
                    
        return hypothesis_id
        
    def update_hypothesis_status(self,
                               hypothesis_id: str,
                               status: str,
                               evidence: Optional[List[Dict]] = None):
        """Update hypothesis status based on findings."""
        valid_statuses = ['proposed', 'testing', 'supported', 'refuted', 'inconclusive']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
            
        with self.driver.session() as session:
            session.run("""
                MATCH (h:Hypothesis {id: $hyp_id})
                SET h.status = $status,
                    h.updated_at = datetime()
            """, hyp_id=hypothesis_id, status=status)
            
            # Add evidence relationships if provided
            if evidence:
                for ev in evidence:
                    self._link_evidence_to_hypothesis(session, hypothesis_id, ev)
                    
    def _link_evidence_to_hypothesis(self, 
                                   session,
                                   hypothesis_id: str,
                                   evidence: Dict):
        """Link evidence to a hypothesis."""
        evidence_id = hashlib.md5(
            f"{evidence.get('source', 'unknown')}_{evidence.get('content', '')[:50]}".encode()
        ).hexdigest()[:16]
        
        session.run("""
            MATCH (h:Hypothesis {id: $hyp_id})
            MERGE (e:Evidence {id: $ev_id})
            ON CREATE SET 
                e.source = $source,
                e.content = $content,
                e.type = $type,
                e.created_at = datetime()
            CREATE (h)-[:SUPPORTED_BY {strength: $strength}]->(e)
        """, hyp_id=hypothesis_id, ev_id=evidence_id,
            source=evidence.get('source', 'unknown'),
            content=evidence.get('content', ''),
            type=evidence.get('type', 'literature'),
            strength=evidence.get('strength', 0.5))
            
    def store_finding(self,
                     conversation_id: str,
                     finding: str,
                     confidence: float,
                     supporting_hypotheses: Optional[List[str]] = None) -> str:
        """Store a research finding."""
        finding_id = hashlib.md5(
            f"{finding}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        with self.driver.session() as session:
            session.run("""
                MATCH (c:Conversation {id: $conv_id})
                CREATE (f:Finding {
                    id: $find_id,
                    content: $content,
                    confidence: $confidence,
                    created_at: datetime()
                })
                CREATE (c)-[:DISCOVERED]->(f)
            """, conv_id=conversation_id, find_id=finding_id,
                content=finding, confidence=confidence)
            
            # Link to supporting hypotheses
            if supporting_hypotheses:
                for hyp_id in supporting_hypotheses:
                    session.run("""
                        MATCH (f:Finding {id: $find_id})
                        MATCH (h:Hypothesis {id: $hyp_id})
                        CREATE (h)-[:LED_TO]->(f)
                    """, find_id=finding_id, hyp_id=hyp_id)
                    
        return finding_id
        
    def get_conversation_context(self, 
                               conversation_id: str,
                               max_messages: int = 10) -> Dict[str, Any]:
        """Get conversation context including entities and hypotheses."""
        with self.driver.session() as session:
            # Get recent messages
            messages_result = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:HAS_MESSAGE]->(m:Message)
                RETURN m.role as role, m.content as content, m.timestamp as timestamp
                ORDER BY m.timestamp DESC
                LIMIT $limit
            """, conv_id=conversation_id, limit=max_messages)
            
            messages = [dict(record) for record in messages_result]
            
            # Get mentioned entities
            entities_result = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:HAS_MESSAGE]->(:Message)-[:MENTIONS]->(e:Entity)
                RETURN DISTINCT e.name as name, e.type as type, count(*) as mentions
                ORDER BY mentions DESC
                LIMIT 20
            """, conv_id=conversation_id)
            
            entities = [dict(record) for record in entities_result]
            
            # Get hypotheses
            hypotheses_result = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:GENERATED]->(h:Hypothesis)
                RETURN h.id as id, h.content as content, h.status as status, h.confidence as confidence
                ORDER BY h.created_at DESC
                LIMIT 10
            """, conv_id=conversation_id)
            
            hypotheses = [dict(record) for record in hypotheses_result]
            
            # Get findings
            findings_result = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:DISCOVERED]->(f:Finding)
                RETURN f.content as content, f.confidence as confidence
                ORDER BY f.confidence DESC
                LIMIT 10
            """, conv_id=conversation_id)
            
            findings = [dict(record) for record in findings_result]
            
            return {
                "conversation_id": conversation_id,
                "messages": messages,
                "entities": entities,
                "hypotheses": hypotheses,
                "findings": findings
            }
            
    def find_related_research(self,
                            entity_names: List[str],
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Find related research based on entities."""
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $entities as entity_name
                MATCH (e:Entity {name: entity_name})<-[:MENTIONS|INVOLVES]-(item)
                WHERE item:Hypothesis OR item:Finding
                WITH item, collect(DISTINCT e.name) as related_entities
                RETURN 
                    labels(item)[0] as type,
                    item.content as content,
                    item.confidence as confidence,
                    related_entities
                ORDER BY item.confidence DESC
                LIMIT $limit
            """, entities=entity_names, limit=limit)
            
            return [dict(record) for record in result]
            
    def get_entity_graph(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get the relationship graph around an entity."""
        with self.driver.session() as session:
            # Get connected entities
            result = session.run("""
                MATCH path = (e1:Entity {name: $name})-[*1..$depth]-(e2:Entity)
                WITH e1, e2, path
                RETURN DISTINCT
                    e2.name as connected_entity,
                    e2.type as entity_type,
                    length(path) as distance,
                    [rel in relationships(path) | type(rel)] as relationship_chain
                ORDER BY distance, connected_entity
                LIMIT 50
            """, name=entity_name, depth=depth)
            
            connections = [dict(record) for record in result]
            
            # Get hypotheses involving this entity
            hyp_result = session.run("""
                MATCH (e:Entity {name: $name})<-[:INVOLVES]-(h:Hypothesis)
                RETURN h.content as hypothesis, h.status as status, h.confidence as confidence
                ORDER BY h.confidence DESC
                LIMIT 10
            """, name=entity_name)
            
            hypotheses = [dict(record) for record in hyp_result]
            
            return {
                "entity": entity_name,
                "connections": connections,
                "hypotheses": hypotheses
            }
            
    def suggest_next_research(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Suggest next research directions based on conversation history."""
        with self.driver.session() as session:
            # Find unexplored connections
            unexplored = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:HAS_MESSAGE]->(:Message)-[:MENTIONS]->(e1:Entity)
                MATCH (e1)-[:RELATED_TO]-(e2:Entity)
                WHERE NOT EXISTS((c)-[:HAS_MESSAGE]->(:Message)-[:MENTIONS]->(e2))
                RETURN DISTINCT e2.name as entity, e2.type as type, count(*) as connection_strength
                ORDER BY connection_strength DESC
                LIMIT 5
            """, conv_id=conversation_id)
            
            suggestions = []
            for record in unexplored:
                suggestions.append({
                    "type": "explore_entity",
                    "entity": record["entity"],
                    "entity_type": record["type"],
                    "reason": f"Connected to {record['connection_strength']} mentioned entities"
                })
                
            # Find hypotheses that need testing
            untested = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:GENERATED]->(h:Hypothesis {status: 'proposed'})
                RETURN h.id as id, h.content as content, h.confidence as confidence
                ORDER BY h.confidence DESC
                LIMIT 3
            """, conv_id=conversation_id)
            
            for record in untested:
                suggestions.append({
                    "type": "test_hypothesis",
                    "hypothesis_id": record["id"],
                    "content": record["content"],
                    "reason": "Hypothesis not yet tested"
                })
                
            return suggestions
            
    def export_research_graph(self, conversation_id: str) -> Dict[str, Any]:
        """Export the entire research graph for a conversation."""
        with self.driver.session() as session:
            # Get all nodes and relationships
            result = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[r*]-(n)
                WITH collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships
                RETURN nodes, relationships
            """, conv_id=conversation_id)
            
            record = result.single()
            if not record:
                return {"nodes": [], "edges": []}
                
            nodes = []
            for node in record["nodes"]:
                node_data = dict(node)
                node_data["labels"] = list(node.labels)
                nodes.append(node_data)
                
            edges = []
            for rel in record["relationships"]:
                edges.append({
                    "type": type(rel).__name__,
                    "properties": dict(rel)
                })
                
            return {
                "nodes": nodes,
                "edges": edges,
                "exported_at": datetime.now().isoformat()
            } 