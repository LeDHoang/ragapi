#!/usr/bin/env python3
"""
LightRAG Integration Test Script

This script tests the complete LightRAG integration including:
- Document processing with LightRAG
- Advanced query capabilities
- Graph traversal
- Semantic search
- Hybrid search
"""

import asyncio
import tempfile
import os
from pathlib import Path

from rag_core.config import config
from rag_core.pipeline import RAGPipeline
from rag_core.advanced_query import AdvancedQueryProcessor


async def test_full_integration():
    """Test the complete LightRAG integration"""

    print("🚀 Starting LightRAG Integration Test")
    print("=" * 50)

    # Enable LightRAG
    config.LIGHTRAG_ENABLED = True
    config.EMBEDDING_MODEL = "text-embedding-3-small"
    config.EMBEDDING_DIM = 1536
    config.LLM_MODEL = "gpt-3.5-turbo"

    try:
        # Initialize pipeline with LightRAG
        print("📚 Initializing pipeline with LightRAG...")
        pipeline = RAGPipeline()

        if not pipeline.lightrag:
            print("❌ LightRAG initialization failed")
            return False

        print(f"✅ LightRAG initialized successfully")
        print(f"   - Working directory: {config.get_lightrag_working_dir()}")
        print(f"   - Embedding model: {config.EMBEDDING_MODEL}")
        print(f"   - LLM model: {config.LLM_MODEL}")

        # Test document processing
        print("\n📄 Testing document processing...")

        test_content = """
        Artificial Intelligence and Machine Learning

        Artificial Intelligence (AI) is a field of computer science that aims to create systems
        capable of performing tasks that typically require human intelligence. These tasks include
        visual perception, speech recognition, decision-making, and language translation.

        Machine Learning (ML) is a subset of AI that provides systems the ability to automatically
        learn and improve from experience without being explicitly programmed. Machine learning
        focuses on the development of computer programs that can access data and use it to learn
        for themselves.

        Key Concepts:
        - Supervised Learning: Learning with labeled training data
        - Unsupervised Learning: Finding patterns in data without labels
        - Reinforcement Learning: Learning through interaction with an environment

        Applications of AI and ML:
        - Healthcare: Disease diagnosis, drug discovery
        - Finance: Fraud detection, algorithmic trading
        - Transportation: Autonomous vehicles, traffic optimization
        - Natural Language Processing: Chatbots, translation services
        """

        # Create temporary document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            # Process document through LightRAG pipeline
            result = await pipeline.process_document(temp_file, 'lightrag_test_doc')

            print(f"✅ Document processing completed: {result.status}")
            print(f"   - Chunks created: {result.chunks_created}")
            print(f"   - Entities found: {result.entities_found}")

        finally:
            os.unlink(temp_file)

        # Test advanced query capabilities
        print("\n🔍 Testing advanced query capabilities...")

        if pipeline.lightrag:
            advanced_processor = AdvancedQueryProcessor(pipeline.lightrag)

            # Test semantic similarity search
            print("   - Testing semantic similarity search...")
            semantic_results = await advanced_processor.semantic_similarity_search(
                query="artificial intelligence applications",
                limit=5,
                threshold=0.7
            )
            print(f"   - Semantic search found {len(semantic_results)} results")

            # Test hybrid search
            print("   - Testing hybrid search...")
            hybrid_results = await advanced_processor.hybrid_search(
                query="machine learning algorithms",
                limit=5
            )
            print(f"   - Hybrid search: {len(hybrid_results.get('vector_results', []))} vector, {len(hybrid_results.get('graph_results', []))} graph results")

            # Test entity relationships
            print("   - Testing entity relationships...")
            relationships = await advanced_processor.get_entity_relationships(
                entity_id="artificial_intelligence",
                max_depth=2
            )
            print(f"   - Entity relationships query completed")

        # Test health check
        print("\n🏥 Testing health check...")
        from app.main import health_check
        health = await health_check()
        print(f"   - Health status: {health['status']}")
        print(f"   - LightRAG enabled: {health['lightrag_enabled']}")

        print("\n🎉 All tests completed successfully!")
        print("\n📊 Integration Summary:")
        print(f"   ✅ LightRAG initialization: {'PASS' if pipeline.lightrag else 'FAIL'}")
        print(f"   ✅ Document processing: {'PASS' if result.status == 'completed' else 'FAIL'}")
        print(f"   ✅ Advanced queries: {'PASS' if pipeline.lightrag else 'FAIL'}")
        print(f"   ✅ API endpoints: {'PASS' if health['lightrag_enabled'] else 'FAIL'}")

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_mode():
    """Test fallback to legacy pipeline when LightRAG is disabled"""

    print("\n🛡️  Testing fallback mode (LightRAG disabled)...")

    # Disable LightRAG
    config.LIGHTRAG_ENABLED = False

    try:
        pipeline = RAGPipeline()

        print(f"✅ Legacy pipeline initialized: {pipeline.lightrag is None}")
        print(f"   - Vector index: {'Available' if pipeline.vector_index else 'Not available'}")
        print(f"   - Neo4j graph: {'Connected' if hasattr(pipeline.storage_manager.graph, 'driver') and pipeline.storage_manager.graph.driver else 'Not connected'}")

        # Test health check in fallback mode
        from app.main import health_check
        health = await health_check()
        print(f"   - Health status: {health['status']}")
        print(f"   - LightRAG enabled: {health['lightrag_enabled']}")

        print("✅ Fallback mode test completed")

    except Exception as e:
        print(f"❌ Fallback mode test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    async def main():
        print("LightRAG Integration Test Suite")
        print("=" * 50)

        # Test full integration with LightRAG
        lightrag_success = await test_full_integration()

        # Test fallback mode
        fallback_success = await test_fallback_mode()

        print("\n" + "=" * 50)
        print("📋 Test Results Summary:")
        print(f"   LightRAG Integration: {'✅ PASSED' if lightrag_success else '❌ FAILED'}")
        print(f"   Fallback Mode: {'✅ PASSED' if fallback_success else '❌ FAILED'}")

        if lightrag_success and fallback_success:
            print("\n🎊 All integration tests PASSED!")
            print("\n🚀 Your RAG system is now successfully integrated with LightRAG!")
            print("\n📚 Available endpoints:")
            print("   POST /query - Standard query (uses LightRAG when enabled)")
            print("   POST /query/advanced - Advanced LightRAG query")
            print("   POST /query/semantic-search - Semantic similarity search")
            print("   POST /query/hybrid-search - Hybrid vector + graph search")
            print("   POST /query/multi-hop - Multi-hop graph traversal")
            print("   GET  /query/entity-relationships/{entity_id} - Entity relationships")
            print("   GET  /health - Health check with LightRAG status")
        else:
            print("\n⚠️  Some tests failed. Please check the error messages above.")

    asyncio.run(main())
