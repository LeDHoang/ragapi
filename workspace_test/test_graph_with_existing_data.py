#!/usr/bin/env python3

import asyncio
import os
from rag_core.storage import StorageManager
from rag_core.pipeline import RAGPipeline
from rag_core.query import QueryProcessor

async def test_graph_with_existing_data():
    """Test graph features with existing processed data"""
    print("🔍 Testing graph features with existing processed data...")

    try:
        # Initialize storage manager
        sm = StorageManager()
        print(f"✅ Graph driver available: {sm.graph.driver is not None}")

        if sm.graph.driver:
            print("✅ Neo4j connected successfully")

            # Check what documents are already processed
            print("\n📋 Checking existing documents...")
            from rag_core.config import config
            working_dir = config.get_working_dir()
            metadata_dir = working_dir / "kv" / "metadata"

            if metadata_dir.exists():
                import json
                processed_docs = []

                for metadata_file in metadata_dir.glob("*.json"):
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            processed_docs.append(metadata)
                    except Exception as e:
                        print(f"   ⚠️  Error reading {metadata_file}: {e}")

                print(f"✅ Found {len(processed_docs)} processed documents")

                if processed_docs:
                    print("\n📄 Document details:")
                    for i, doc in enumerate(processed_docs[:3]):  # Show first 3
                        print(f"   {i+1}. {doc.get('doc_id', 'Unknown')}")
                        print(f"      Path: {doc.get('file_path', 'Unknown')}")
                        print(f"      Pages: {doc.get('total_pages', 0)}")
                        print(f"      Chunks: {doc.get('chunks_count', 0)}")
                        print(f"      Entities: {doc.get('entities_count', 0)}")
                        print()

            # Test entity extraction from existing chunks
            print("🔄 Testing entity extraction from existing content...")

            # Read some existing chunks
            chunks_dir = config.get_working_dir() / "text_chunks"
            if chunks_dir.exists():
                chunk_files = list(chunks_dir.glob("*.txt"))
                print(f"✅ Found {len(chunk_files)} existing chunks")

                if chunk_files:
                    # Test entity extraction on first chunk
                    import json
                    with open(chunk_files[0]) as f:
                        chunk_data = json.load(f)

                    chunk_content = chunk_data.get('content', '')
                    chunk_id = chunk_data.get('id', 'unknown')
                    doc_id = chunk_data.get('doc_id', 'unknown')

                    print(f"   Processing chunk: {chunk_id}")
                    print(f"   Content length: {len(chunk_content)} characters")

                    # Extract entities using fallback method (no LLM needed)
                    entities = await sm._fallback_entity_extraction(
                        content=chunk_content,
                        chunk_id=chunk_id,
                        file_path=chunk_data.get('file_path', '/unknown')
                    )

                    print(f"✅ Extracted {len(entities)} entities from chunk")

                    if entities:
                        print("   📊 Sample entities:")
                        for entity in entities[:3]:
                            print(f"      - {entity.name} ({entity.entity_type})")

            # Test graph queries
            print("\n🔍 Testing graph queries...")

            # Get all entities from graph
            query = """
            MATCH (e:Entity)
            RETURN e.name, e.entity_type, e.description
            LIMIT 10
            """

            with sm.graph.driver.session() as session:
                result = session.run(query)
                entities_in_graph = []
                for record in result:
                    entities_in_graph.append({
                        "name": record["e.name"],
                        "type": record["e.entity_type"],
                        "description": record["e.description"]
                    })

            print(f"✅ Found {len(entities_in_graph)} entities in graph database")

            if entities_in_graph:
                print("   📊 Entities in graph:")
                for i, entity in enumerate(entities_in_graph[:5]):
                    print(f"      {i+1}. {entity['name']} ({entity['type']})")
                    print(f"         {entity['description'][:100]}...")

            # Test relationships
            query = """
            MATCH ()-[r]-()
            RETURN type(r) as relationship_type, count(r) as count
            """

            with sm.graph.driver.session() as session:
                result = session.run(query)
                relationships = []
                for record in result:
                    relationships.append({
                        "type": record["relationship_type"],
                        "count": record["count"]
                    })

            print(f"\n✅ Found {len(relationships)} relationship types in graph")
            for rel in relationships:
                print(f"   - {rel['type']}: {rel['count']} relationships")

            # Test query processor with graph enhancement
            print("\n🤖 Testing graph-enhanced query processing...")

            processor = QueryProcessor()

            # Test entity extraction from query
            test_query = "What are the key concepts in machine learning?"
            query_entities = await processor._extract_entities_from_query(test_query)

            print(f"✅ Query: '{test_query}'")
            print(f"✅ Extracted entities: {query_entities}")

            # Test graph context retrieval
            if query_entities:
                graph_context = await processor._get_graph_context(query_entities)
                if graph_context:
                    print(f"✅ Graph context found: {graph_context[:200]}...")
                else:
                    print("   ℹ️  No graph context found (may need LLM for entity matching)")

            print("\n🎉 Graph integration test with existing data completed successfully!")
            print("\n📈 Graph Statistics:")
            print(f"   - Documents in graph: {len(processed_docs) if 'processed_docs' in locals() else 'N/A'}")
            print(f"   - Entities in graph: {len(entities_in_graph)}")
            print(f"   - Relationships in graph: {sum(r['count'] for r in relationships)}")

        else:
            print("❌ Neo4j connection failed - make sure Neo4j is running")
            print("   Run: neo4j start")

    except Exception as e:
        print(f"❌ Graph test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_graph_with_existing_data())
