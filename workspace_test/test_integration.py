#!/usr/bin/env python3

import asyncio
from rag_core.storage import StorageManager
from rag_core.schemas import EntityNode, EntityRelation

async def test_integration():
    """Test the full graph integration"""
    print("Testing full graph integration...")

    try:
        sm = StorageManager()
        print(f"✅ Graph driver available: {sm.graph.driver is not None}")

        if sm.graph.driver:
            print("✅ Neo4j connected successfully")

            # Test document creation
            await sm.store_document(
                doc_id="test_doc_integration",
                file_path="/test/integration.pdf",
                metadata={
                    "file_type": ".pdf",
                    "total_pages": 10,
                    "chunks_count": 20,
                    "entities_count": 15,
                    "processed_at": 1234567890.0
                }
            )
            print("✅ Document stored in graph")

            # Test chunk creation
            await sm.store_chunk(
                chunk_id="test_chunk_123",
                content="This document discusses Machine Learning and Artificial Intelligence concepts. John Smith from OpenAI Corporation developed the API for this ML system.",
                doc_id="test_doc_integration",
                chunk_type="text",
                page_idx=1,
                vector=[0.1, 0.2, 0.3] * 1024  # Mock embedding
            )
            print("✅ Chunk stored in graph")

            # Test entity extraction and storage
            entities = await sm.extract_and_store_entities(
                content="This document discusses Machine Learning and Artificial Intelligence concepts. John Smith from OpenAI Corporation developed the API for this ML system.",
                chunk_id="test_chunk_123",
                doc_id="test_doc_integration",
                file_path="/test/integration.pdf",
                content_type="text"
            )
            print(f"✅ Extracted and stored {len(entities)} entities")

            # Test relationship creation
            await sm.find_entity_relationships("test_doc_integration")
            print("✅ Entity relationships created")

            print("🎉 Full graph integration test successful!")
            print("\n📊 Graph Statistics:")
            # Get some stats from the graph
            query = """
            MATCH (n)
            RETURN labels(n) as labels, count(n) as count
            """
            with sm.graph.driver.session() as session:
                result = session.run(query)
                for record in result:
                    print(f"   {record['labels']}: {record['count']}")

        else:
            print("❌ Neo4j connection failed")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integration())
