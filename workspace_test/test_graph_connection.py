#!/usr/bin/env python3

import asyncio
from rag_core.storage import StorageManager

async def test_graph_connection():
    """Test Neo4j graph database connection"""
    print("Testing Neo4j graph database connection...")

    try:
        sm = StorageManager()
        print(f"Graph driver available: {sm.graph.driver is not None}")

        if sm.graph.driver:
            print("✅ Neo4j connection successful!")

            # Test creating a simple entity
            print("Testing entity creation...")
            from rag_core.schemas import EntityNode
            from rag_core.storage import EntityRelation

            # Create a test document
            await sm.store_document(
                doc_id="test_doc_123",
                file_path="/test/path.pdf",
                metadata={
                    "file_type": ".pdf",
                    "total_pages": 5,
                    "chunks_count": 10,
                    "entities_count": 5,
                    "processed_at": 1234567890.0
                }
            )
            print("✅ Document created in graph")

            # Create a test entity
            entity = EntityNode(
                entity_id="test_entity_123",
                entity_type="concept",
                name="Machine Learning",
                description="A type of artificial intelligence",
                source_id="chunk_123",
                file_path="/test/path.pdf",
                created_at=1234567890.0
            )

            await sm.store_entity(entity)
            print("✅ Entity created in graph")

            # Create a test relationship
            relation = EntityRelation(
                src_id="test_entity_123",
                tgt_id="chunk_123",
                relation_type="APPEARS_ON",
                description="Entity appears in chunk",
                keywords="machine learning,concept",
                source_id="chunk_123",
                weight=0.8,
                file_path="/test/path.pdf"
            )

            await sm.store_relation(relation)
            print("✅ Relationship created in graph")

            print("🎉 All graph operations successful!")

        else:
            print("❌ Neo4j connection failed")

    except Exception as e:
        print(f"❌ Error testing graph connection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_graph_connection())
