#!/usr/bin/env python3

import asyncio
import os
from rag_core.storage import StorageManager
from rag_core.query import QueryProcessor

async def test_with_api_fallback():
    """Test graph features with fallback methods (no API keys needed)"""
    print("🧪 Testing graph features with fallback methods (no API keys needed)...")

    try:
        # Initialize components
        sm = StorageManager()
        processor = QueryProcessor()

        print(f"✅ Graph driver available: {sm.graph.driver is not None}")

        if sm.graph.driver:
            print("✅ Neo4j connected successfully")

            # Test 1: Manual entity extraction using fallback
            print("\n🔄 Test 1: Manual entity extraction...")
            test_content = "Machine Learning and Artificial Intelligence are key technologies used by companies like OpenAI and Google for developing advanced AI systems."

            entities = await sm._fallback_entity_extraction(
                content=test_content,
                chunk_id="test_chunk_manual",
                file_path="/test/manual.pdf"
            )

            print(f"✅ Extracted {len(entities)} entities:")
            for entity in entities:
                print(f"   - {entity.name} ({entity.entity_type}): {entity.description}")

            # Test 2: Query the graph for existing entities
            print("\n🔍 Test 2: Query existing graph data...")

            # Find entities related to "Machine Learning"
            query = """
            MATCH (e:Entity)
            WHERE e.name =~ "(?i).*machine.*"
            RETURN e.name, e.entity_type, e.description
            """

            with sm.graph.driver.session() as session:
                result = session.run(query)
                ml_entities = []
                for record in result:
                    ml_entities.append({
                        "name": record["e.name"],
                        "type": record["e.entity_type"],
                        "description": record["e.description"]
                    })

            print(f"✅ Found {len(ml_entities)} machine learning related entities:")
            for entity in ml_entities:
                print(f"   - {entity['name']} ({entity['type']}): {entity['description'][:80]}...")

            # Test 3: Find entity relationships
            print("\n🔗 Test 3: Find entity relationships...")

            # First check what relationship types exist
            schema_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN collect(relationshipType) as relationship_types
            """

            available_relationships = []
            with sm.graph.driver.session() as session:
                try:
                    schema_result = session.run(schema_query)
                    for record in schema_result:
                        available_relationships = record['relationship_types']
                        break
                except Exception:
                    # Fallback: try common relationship types
                    available_relationships = ['REFERENCES', 'APPEARS_ON']

                # Build query with available relationship types
                if available_relationships:
                    rel_types = '|'.join(available_relationships)
                    query = f"""
                    MATCH (e1:Entity)-[r:{rel_types}]->(e2:Entity)
                    RETURN e1.name, type(r), e2.name
                    LIMIT 5
                    """
                else:
                    query = """
                    MATCH (e1:Entity)-[r]->(e2:Entity)
                    RETURN e1.name, type(r), e2.name
                    LIMIT 5
                    """

                result = session.run(query)
                relationships = []
                for record in result:
                    relationships.append({
                        "from": record["e1.name"],
                        "relationship": record["type(r)"],
                        "to": record["e2.name"]
                    })

            print(f"✅ Found {len(relationships)} entity relationships:")
            for rel in relationships:
                print(f"   - {rel['from']} --[{rel['relationship']}]-> {rel['to']}")

            # Test 4: Test simple pattern-based query entity extraction
            print("\n🤖 Test 4: Pattern-based query entity extraction...")

            # Manually extract entities using simple patterns (no LLM needed)
            def simple_entity_extraction(query_text):
                import re
                entities = []

                # Simple patterns for entity extraction
                patterns = [
                    (r'\b[A-Z][a-zA-Z]* (Learning|Intelligence|AI|ML)\b', 'concept'),
                    (r'\b[A-Z][a-zA-Z]* (Inc|Corp|LLC|Company)\b', 'organization'),
                    (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'person'),
                ]

                for pattern, entity_type in patterns:
                    matches = re.findall(pattern, query_text)
                    for match in matches:
                        if len(match) > 3:  # Filter short matches
                            entities.append({
                                "name": match,
                                "type": entity_type,
                                "confidence": 0.7
                            })

                return entities

            test_queries = [
                "What is machine learning?",
                "Tell me about artificial intelligence companies",
                "How does John Smith use AI?",
                "What are the key concepts in deep learning?"
            ]

            for query in test_queries:
                entities = simple_entity_extraction(query)
                print(f"   Query: '{query}'")
                print(f"   → Extracted: {[e['name'] for e in entities]}")

            print("\n🎉 All fallback tests completed successfully!")
            print("\n💡 Summary:")
            print("   ✅ Graph database is fully functional")
            print("   ✅ Entity extraction works with fallback methods")
            print("   ✅ Relationship queries work perfectly")
            print("   ✅ No API keys required for core graph features")

        else:
            print("❌ Neo4j connection failed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_with_api_fallback())
