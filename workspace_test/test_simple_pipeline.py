#!/usr/bin/env python3

import asyncio
from rag_core.parsers import DoclingParser
from rag_core.pipeline import RAGPipeline
import tempfile
import os

async def test_simple_pipeline():
    """Test pipeline with docling parser directly"""
    print("Testing pipeline with docling parser...")

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

        # Test docling parser directly
        print("🔄 Testing docling parser...")
        parser = DoclingParser()
        content_list = await parser.parse_document(temp_file_path)

        print(f"✅ Docling parser extracted {len(content_list)} content items")

        # Initialize pipeline and test content processing
        print("🔄 Testing pipeline content processing...")
        pipeline = RAGPipeline()

        # Manually separate content (bypass the parser selection)
        from rag_core.processors import ContentSeparator
        separator = ContentSeparator()
        full_text, multimodal_items, summary = await separator.process_document_content(
            content_list, "test_simple_123"
        )

        print(f"✅ Content separated: {len(full_text)} chars text, {len(multimodal_items)} multimodal items")

        # Test text processing
        print("🔄 Processing text content...")
        await pipeline._process_text_content(full_text, "test_simple_123", temp_file_path)

        # Count chunks
        chunks_count = await pipeline._count_chunks("test_simple_123")
        print(f"✅ Created {chunks_count} text chunks")

        # Get entities
        entities = await pipeline._get_entities("test_simple_123")
        print(f"✅ Found {len(entities)} entities in graph")

        if entities:
            print("📊 Sample entities:")
            for i, entity in enumerate(entities[:5]):  # Show first 5 entities
                print(f"   {i+1}. {entity}")

        # Clean up
        os.unlink(temp_file_path)
        print("✅ Cleaned up temporary file")

        print("🎉 Simple pipeline test with graph integration successful!")

    except Exception as e:
        print(f"❌ Simple pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_pipeline())
