#!/usr/bin/env python3
"""
Test script to demonstrate duplicate document detection functionality.
This script shows how the system handles duplicate files and skips reprocessing.
"""

import asyncio
import aiohttp
import json
from pathlib import Path

async def test_duplicate_detection():
    """Test the duplicate detection functionality"""

    base_url = "http://localhost:8001"

    # Test file (assuming we have a test PDF)
    test_file = "input/first5sheets_auto.pdf"

    if not Path(test_file).exists():
        print(f"❌ Test file {test_file} not found. Please ensure the file exists.")
        return

    print("🧪 Testing Duplicate Document Detection")
    print("=" * 50)

    # Step 1: Upload the same document twice
    print("\n📤 Step 1: First upload")
    async with aiohttp.ClientSession() as session:
        with open(test_file, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="input/first5sheets_auto.pdf")

            async with session.post(f"{base_url}/ingest", data=data) as resp:
                result = await resp.json()
                print(f"Status: {resp.status}")
                print(f"Response: {json.dumps(result, indent=2)}")

                if resp.status == 200:
                    first_doc_id = result["doc_id"]
                    print(f"✅ First document processed with ID: {first_doc_id}")

    # Wait a moment for processing to complete
    await asyncio.sleep(2)

    print("\n📤 Step 2: Upload the same document again (should be detected as duplicate)")
    async with aiohttp.ClientSession() as session:
        with open(test_file, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="input/first5sheets_auto.pdf")

            async with session.post(f"{base_url}/ingest", data=data) as resp:
                result = await resp.json()
                print(f"Status: {resp.status}")
                print(f"Response: {json.dumps(result, indent=2)}")

                if resp.status == 200:
                    if result.get("is_duplicate"):
                        print(f"✅ Duplicate detected! Returned existing document ID: {result['doc_id']}")
                        print(f"   Notes: {result.get('notes', 'N/A')}")
                    else:
                        print(f"⚠️  Document was processed again (no duplicate detection)")
                        print(f"   New document ID: {result['doc_id']}")

    # Step 3: Upload with different filename but same content
    print("\n📤 Step 3: Upload same content with different filename")
    async with aiohttp.ClientSession() as session:
        with open(test_file, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="different_name.pdf")

            async with session.post(f"{base_url}/ingest", data=data) as resp:
                result = await resp.json()
                print(f"Status: {resp.status}")
                print(f"Response: {json.dumps(result, indent=2)}")

                if resp.status == 200:
                    if result.get("is_duplicate"):
                        print(f"✅ Content-based duplicate detected! Returned existing document ID: {result['doc_id']}")
                        print(f"   Notes: {result.get('notes', 'N/A')}")
                    else:
                        print(f"⚠️  Document was processed again (no content-based duplicate detection)")
                        print(f"   New document ID: {result['doc_id']}")

    # Step 4: List all documents
    print("\n📋 Step 4: List all processed documents")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/documents") as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"Total documents: {result['total']}")
                for doc in result['documents']:
                    print(f"  - {doc['doc_id']}: {doc['filename']} ({doc['total_blocks']} blocks)")
            else:
                print(f"❌ Failed to list documents: {resp.status}")

    # Step 5: Query the documents
    print("\n🔍 Step 5: Test querying the processed documents")
    query_data = {
        "query": "What is the main content of this document?",
        "mode": "text",
        "k": 5
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/query", json=query_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print("✅ Query successful!")
                print(f"Answer preview: {result['answer'][:200]}...")
            else:
                print(f"❌ Query failed: {resp.status}")
                print(f"Response: {await resp.text()}")

    print("\n🎉 Duplicate detection test completed!")

if __name__ == "__main__":
    # Note: Make sure the server is running before running this test
    print("Make sure the RAG API server is running on http://localhost:8000")
    print("Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("\nThen run this test script.\n")

    asyncio.run(test_duplicate_detection())
