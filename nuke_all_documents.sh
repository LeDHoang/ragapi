#!/bin/bash

# Script to delete all documents from the RAG server
# Usage: ./nuke_all_documents.sh [server_url]

SERVER_URL=${1:-"http://localhost:8001"}

echo "🚀 Starting document deletion process..."
echo "Server URL: $SERVER_URL"

# Get all documents
echo "📋 Fetching list of documents..."
DOCUMENTS_RESPONSE=$(curl -s -X GET "$SERVER_URL/documents")

if [ $? -ne 0 ]; then
    echo "❌ Failed to connect to server at $SERVER_URL"
    exit 1
fi

# Extract document IDs using jq (if available) or basic parsing
if command -v jq &> /dev/null; then
    # Use jq for proper JSON parsing
    DOC_IDS=$(echo "$DOCUMENTS_RESPONSE" | jq -r '.documents[].doc_id')
    TOTAL_COUNT=$(echo "$DOCUMENTS_RESPONSE" | jq -r '.total')
else
    # Fallback: basic grep parsing (less reliable)
    DOC_IDS=$(echo "$DOCUMENTS_RESPONSE" | grep -o '"doc_id":"[^"]*"' | sed 's/"doc_id":"\([^"]*\)"/\1/')
    TOTAL_COUNT=$(echo "$DOCUMENTS_RESPONSE" | grep -o '"total":[0-9]*' | sed 's/"total"://')
fi

if [ -z "$DOC_IDS" ]; then
    echo "✅ No documents found to delete"
    exit 0
fi

echo "📊 Found $TOTAL_COUNT documents to delete"

# Delete each document
DELETED_COUNT=0
FAILED_COUNT=0

for doc_id in $DOC_IDS; do
    echo "🗑️  Deleting document: $doc_id"
    
    DELETE_RESPONSE=$(curl -s -X DELETE "$SERVER_URL/documents/$doc_id")
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully deleted: $doc_id"
        ((DELETED_COUNT++))
    else
        echo "❌ Failed to delete: $doc_id"
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "📈 Deletion Summary:"
echo "   ✅ Successfully deleted: $DELETED_COUNT documents"
echo "   ❌ Failed to delete: $FAILED_COUNT documents"
echo "   📊 Total processed: $((DELETED_COUNT + FAILED_COUNT)) documents"

# Verify deletion
echo ""
echo "🔍 Verifying deletion..."
FINAL_RESPONSE=$(curl -s -X GET "$SERVER_URL/documents")
if command -v jq &> /dev/null; then
    REMAINING_COUNT=$(echo "$FINAL_RESPONSE" | jq -r '.total')
else
    REMAINING_COUNT=$(echo "$FINAL_RESPONSE" | grep -o '"total":[0-9]*' | sed 's/"total"://')
fi

if [ "$REMAINING_COUNT" = "0" ]; then
    echo "🎉 All documents successfully deleted!"
else
    echo "⚠️  Warning: $REMAINING_COUNT documents still remain"
fi

echo "🏁 Document deletion process completed"
