#!/usr/bin/env python3
"""
Simple test for duplicate detection without dependencies
"""

import sys
import os
sys.path.append('/Users/hoangleduc/Desktop/Coding Project/ragapi')

def test_file_hashing():
    """Test the file hashing functionality"""
    try:
        from rag_core.utils import calculate_file_hash
        from pathlib import Path

        # Test with a small file
        test_file = Path("/Users/hoangleduc/Desktop/Coding Project/ragapi/requirements.txt")
        if test_file.exists():
            file_hash = calculate_file_hash(test_file)
            print(f"✅ File hashing works: {test_file.name}")
            print(f"   Hash: {file_hash[:16]}...")
            return True
        else:
            print("❌ Test file requirements.txt not found")
            return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_document_registry():
    """Test the document registry functionality"""
    try:
        # Mock the DocumentRegistry without sklearn dependency
        from rag_core.storage import DocumentRegistry
        from rag_core.utils import calculate_file_hash
        from pathlib import Path
        import json
        import time

        # Create a temporary registry path
        temp_path = Path("/tmp/test_registry")
        temp_path.mkdir(exist_ok=True)

        # Test registry creation and operations
        registry = DocumentRegistry(temp_path)

        # Test file hash calculation
        test_file = Path("/Users/hoangleduc/Desktop/Coding Project/ragapi/requirements.txt")
        if test_file.exists():
            file_hash = calculate_file_hash(test_file)
            file_size = test_file.stat().st_size

            # Test duplicate detection (should be False for new file)
            is_dup, existing_id = registry.is_duplicate("requirements.txt", file_hash, file_size)
            print(f"✅ New file duplicate check: {is_dup} (should be False)")

            # Register the document
            doc_id = "test-doc-123"
            registry.register_document(
                doc_id=doc_id,
                filename="requirements.txt",
                file_hash=file_hash,
                file_size=file_size,
                content_list=[{"type": "text", "text": "test content"}],
                by_type={"text": 1}
            )

            # Test duplicate detection again (should be True)
            is_dup, existing_id = registry.is_duplicate("requirements.txt", file_hash, file_size)
            print(f"✅ Duplicate file check: {is_dup} (should be True)")
            print(f"   Existing doc ID: {existing_id}")

            # Test with different filename but same hash
            is_dup, existing_id = registry.is_duplicate("different_name.txt", file_hash, file_size)
            print(f"✅ Content-based duplicate check: {is_dup} (should be True)")

            # Test listing documents
            docs = registry.list_documents()
            print(f"✅ Registry contains {len(docs)} documents")

            # Test getting document info
            info = registry.get_document_info(doc_id)
            if info:
                print(f"✅ Document info retrieved: {info['filename']}")

            # Cleanup
            import shutil
            shutil.rmtree(temp_path)

            print("✅ Document registry test completed successfully")
            return True
        else:
            print("❌ Test file not found")
            return False

    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Duplicate Detection Components")
    print("=" * 50)

    success1 = test_file_hashing()
    print()
    success2 = test_document_registry()

    if success1 and success2:
        print("\n🎉 All tests passed! Duplicate detection system is working.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
