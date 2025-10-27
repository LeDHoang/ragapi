#!/usr/bin/env python3

import asyncio
from rag_core.pipeline import RAGPipeline
import tempfile
import os

async def test_full_pipeline():
    """Test the full pipeline with graph integration"""
    print("Testing full pipeline with graph integration...")

    try:
        # Create a sample text file for testing
        sample_content = """
        # Machine Learning Overview

        Machine Learning is a subset of Artificial Intelligence (AI) that enables systems to learn and improve from experience without being explicitly programmed.

        ## Key Concepts

        ### Neural Networks
        Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process and transmit information.

        ### Deep Learning
        Deep Learning is a subset of Machine Learning that uses multiple layers of neural networks to model complex patterns in data.

        ## Applications

        Machine Learning has numerous applications including:
        - Computer Vision: Image recognition and processing
        - Natural Language Processing: Text analysis and generation
        - Recommendation Systems: Product and content suggestions

        ## Companies and Organizations

        Leading companies in AI include:
        - OpenAI: Research organization focused on AI safety
        - Google: Developer of TensorFlow and other ML tools
        - Microsoft: Azure Machine Learning platform
        - Meta: AI research in computer vision and NLP

        ## Technical Terms

        Important technical terms include:
        - API (Application Programming Interface)
        - SDK (Software Development Kit)
        - ML (Machine Learning)
        - AI (Artificial Intelligence)
        - NLP (Natural Language Processing)
        """

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_content)
            temp_file_path = f.name

        print(f"✅ Created temporary test file: {temp_file_path}")

        # Initialize pipeline
        pipeline = RAGPipeline()
        print("✅ Pipeline initialized")

        # Process document
        print("🔄 Processing document...")
        status = await pipeline.process_document(
            file_path=temp_file_path,
            task_id="test_full_pipeline_123"
        )

        print(f"✅ Document processing completed with status: {status.status}")
        print(f"   Progress: {status.progress}")
        print(f"   Chunks created: {status.chunks_created}")
        print(f"   Entities found: {status.entities_found}")

        # Check if entities were extracted
        entities = await pipeline._get_entities("test_full_pipeline_123")
        print(f"✅ Retrieved {len(entities)} entities from graph")

        if entities:
            print("📊 Sample entities:")
            for i, entity in enumerate(entities[:5]):  # Show first 5 entities
                print(f"   {i+1}. {entity}")

        # Clean up
        os.unlink(temp_file_path)
        print("✅ Cleaned up temporary file")

        print("🎉 Full pipeline test with graph integration successful!")

    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
