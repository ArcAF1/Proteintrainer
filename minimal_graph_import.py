#!/usr/bin/env python3
"""
Minimal Graph Import - Just the essentials for supplement research
Low effort, high impact approach
"""
import asyncio
from src.biomedical_api_client import BiomedicalAPIClient
from neo4j import AsyncGraphDatabase

# Just the compounds that actually matter for workout/supplement research
CORE_COMPOUNDS = {
    "creatine": {
        "effects": ["ATP regeneration", "muscle strength", "cognitive function"],
        "dose": "3-5g/day",
        "timing": "anytime"
    },
    "beta-alanine": {
        "effects": ["muscle endurance", "pH buffering", "reduced fatigue"],
        "dose": "2-5g/day",
        "timing": "divided doses"
    },
    "citrulline": {
        "effects": ["nitric oxide", "blood flow", "reduced soreness"],
        "dose": "6-8g/day", 
        "timing": "pre-workout"
    },
    "caffeine": {
        "effects": ["alertness", "fat oxidation", "power output"],
        "dose": "100-400mg",
        "timing": "pre-workout"
    }
}

async def create_minimal_supplement_graph():
    """Create a minimal but useful graph for supplement research."""
    
    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", 
                                     auth=("neo4j", "password"))
    
    async with driver.session() as session:
        print("🔨 Creating minimal supplement knowledge graph...")
        
        # 1. Create compound nodes with practical info
        for compound, info in CORE_COMPOUNDS.items():
            print(f"  Adding {compound}...")
            
            await session.run("""
                MERGE (c:Compound {name: $name})
                SET c.recommended_dose = $dose,
                    c.timing = $timing,
                    c.updated = datetime()
            """, name=compound, dose=info['dose'], timing=info['timing'])
            
            # Link to effects
            for effect in info['effects']:
                await session.run("""
                    MATCH (c:Compound {name: $compound})
                    MERGE (e:Effect {name: $effect})
                    MERGE (c)-[:PROVIDES]->(e)
                """, compound=compound, effect=effect)
        
        # 2. Add some key interactions/synergies
        print("  Adding synergies...")
        
        # Creatine + Carbs = Better absorption
        await session.run("""
            MATCH (c:Compound {name: 'creatine'})
            CREATE (s:Synergy {
                name: 'Creatine + Carbohydrates',
                mechanism: 'Insulin enhances creatine uptake',
                recommendation: 'Take with 50g carbs for 20% better absorption'
            })
            MERGE (c)-[:SYNERGIZES_WITH]->(s)
        """)
        
        # Beta-alanine + Creatine = Popular stack
        await session.run("""
            MATCH (c1:Compound {name: 'creatine'})
            MATCH (c2:Compound {name: 'beta-alanine'})
            MERGE (c1)-[:STACKS_WELL_WITH]->(c2)
        """)
        
        # 3. Add practical timing relationships
        print("  Adding timing guidance...")
        
        await session.run("""
            CREATE (t:Timing {name: 'Pre-Workout Window', minutes_before: 30})
            WITH t
            MATCH (c:Compound) WHERE c.timing CONTAINS 'pre-workout'
            MERGE (c)-[:BEST_TAKEN_DURING]->(t)
        """)
        
        # 4. Get stats
        result = await session.run("""
            MATCH (n) RETURN count(n) as nodes
            UNION
            MATCH ()-[r]->() RETURN count(r) as relationships
        """)
        
        stats = [record async for record in result]
        print(f"\n✅ Created minimal graph:")
        print(f"   Nodes: {stats[0]['nodes']}")
        print(f"   Relationships: {stats[1]['relationships']}")
        
    await driver.close()

async def query_practical_info(question: str):
    """Example queries that actually help."""
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", 
                                     auth=("neo4j", "password"))
    
    async with driver.session() as session:
        if "stack" in question.lower():
            # What stacks well together?
            result = await session.run("""
                MATCH (c1:Compound)-[:STACKS_WELL_WITH]->(c2:Compound)
                RETURN c1.name + ' + ' + c2.name as stack
            """)
            
        elif "pre-workout" in question.lower():
            # What should I take pre-workout?
            result = await session.run("""
                MATCH (c:Compound)-[:BEST_TAKEN_DURING]->(t:Timing)
                WHERE t.name CONTAINS 'Pre-Workout'
                RETURN c.name as compound, c.recommended_dose as dose
            """)
            
        elif "effects" in question.lower():
            # What are the effects of X?
            result = await session.run("""
                MATCH (c:Compound)-[:PROVIDES]->(e:Effect)
                RETURN c.name as compound, collect(e.name) as effects
            """)
        
        # Return results
        return [record async for record in result]
    
    await driver.close()

if __name__ == "__main__":
    print("Minimal Neo4j Import - Just What Matters\n")
    
    # Create the graph
    asyncio.run(create_minimal_supplement_graph())
    
    print("\nExample queries you can now answer:")
    print("- What stacks well together?")
    print("- What should I take pre-workout?") 
    print("- What are the effects of each compound?")
    
    # Example query
    print("\n🔍 Testing a query...")
    results = asyncio.run(query_practical_info("pre-workout"))
    print("Pre-workout supplements:")
    for r in results:
        print(f"  - {r['compound']}: {r['dose']}") 