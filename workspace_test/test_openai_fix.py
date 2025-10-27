#!/usr/bin/env python3
"""
Test script to verify OpenAI API configuration fix
"""

import asyncio
import os
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from rag_core.config import config
from rag_core.pipeline import RAGPipeline

async def test_openai_configuration():
    """Test OpenAI configuration and LightRAG initialization"""
    
    print("🔧 Testing OpenAI Configuration Fix")
    print("=" * 50)
    
    # Check configuration loading
    print(f"✅ OpenAI API Key: {'Set' if config.OPENAI_API_KEY else 'Not Set'}")
    print(f"✅ Base URL: {config.OPENAI_BASE_URL}")
    print(f"✅ Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"✅ LLM Model: {config.LLM_MODEL}")
    
    # Check model config method
    model_config = config.get_model_config()
    print(f"✅ Model Config - Embedding: {model_config['embedding_model']}")
    print(f"✅ Model Config - LLM: {model_config['llm_model']}")
    
    # Test LightRAG initialization
    print("\n🚀 Testing LightRAG Initialization...")
    try:
        pipeline = RAGPipeline()
        
        if pipeline.lightrag:
            print("✅ LightRAG initialized successfully!")
            print(f"   - Working directory: {config.get_lightrag_working_dir()}")
            print(f"   - Embedding model: {model_config['embedding_model']}")
            print(f"   - LLM model: {model_config['llm_model']}")
        else:
            print("❌ LightRAG initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ LightRAG initialization error: {e}")
        return False
    
    # Test a simple embedding call
    print("\n🧪 Testing Embedding Generation...")
    try:
        from lightrag.llm.openai import openai_embed
        
        # Test with a simple text
        test_text = "This is a test document for embedding generation."
        embeddings = await openai_embed([test_text])
        
        if embeddings and len(embeddings) > 0:
            print(f"✅ Embedding generation successful!")
            print(f"   - Generated {len(embeddings)} embeddings")
            print(f"   - Embedding dimension: {len(embeddings[0])}")
        else:
            print("❌ Embedding generation failed - no embeddings returned")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation error: {e}")
        return False
    
    print("\n🎉 All tests passed! OpenAI configuration is working correctly.")
    return True

if __name__ == "__main__":
    asyncio.run(test_openai_configuration())
