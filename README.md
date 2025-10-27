# 🚀 RAG-Anything FastAPI

A powerful multimodal RAG (Retrieval-Augmented Generation) API built with FastAPI that can process documents containing text, images, tables, and equations. Built on top of LightRAG for advanced knowledge graph capabilities.

## ✨ Features

- **📄 Multimodal Document Processing**: Handle PDF, Word, Excel, PowerPoint, and image files
- **🖼️ Image Analysis**: Extract and analyze visual content using vision models
- **📊 Table Processing**: Intelligent table data extraction and analysis
- **🔬 Equation Understanding**: Process mathematical equations and formulas
- **🕸️ Knowledge Graph**: Advanced graph-based retrieval with LightRAG
- **⚡ FastAPI**: High-performance async API with automatic documentation
- **🔄 Background Processing**: Asynchronous document processing with status tracking
- **💾 Persistent Storage**: Vector databases and knowledge graphs for efficient retrieval

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose (recommended)
- OpenAI API key
- At least 4GB RAM recommended

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ragapi
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   ```

3. **Edit the `.env` file** and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

5. **Test the API**
   ```bash
   curl http://localhost:8000/health
   ```

### Manual Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies** (Ubuntu/Debian)
   ```bash
   sudo apt-get update
   sudo apt-get install libreoffice libreoffice-writer libreoffice-calc libreoffice-impress poppler-utils
   ```

4. **Set up environment**
   ```bash
   cp env.example .env
   # Edit .env file with your configuration
   ```

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 📚 API Documentation

Once the server is running, visit:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔌 API Endpoints

### Document Management

#### Upload Document
```bash
POST /ingest/upload
Content-Type: multipart/form-data

Parameters:
- file: The document file (PDF, DOCX, XLSX, PPTX, images)
- enable_images: Process images in documents (default: true)
- enable_tables: Process tables in documents (default: true)
- enable_equations: Process equations in documents (default: true)
- parser: Document parser to use (docling or mineru, default: docling)

Response:
{
  "task_id": "uuid",
  "status": "processing",
  "message": "Document upload successful, processing started"
}
```

#### Check Processing Status
```bash
GET /ingest/status/{task_id}

Response:
{
  "status": "completed|processing|failed",
  "progress": 1.0,
  "doc_id": "document_id",
  "chunks_created": 150,
  "entities_found": 45
}
```

#### List Documents
```bash
GET /documents

Response:
{
  "documents": [
    {
      "id": "doc_id",
      "file_path": "/path/to/file.pdf",
      "status": "completed",
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 1
}
```

#### Delete Document
```bash
DELETE /documents/{doc_id}

Response:
{
  "message": "Document deleted successfully",
  "doc_id": "deleted_doc_id"
}
```

### Query Interface

#### Basic Query
```bash
POST /query
Content-Type: application/json

{
  "query": "What are the key findings in the document?",
  "query_type": "text",
  "mode": "hybrid"
}

Response:
{
  "result": "Based on the document analysis...",
  "query_type": "text",
  "processing_time": 1.23,
  "entities_found": ["entity1", "entity2"],
  "multimodal_context": []
}
```

#### Advanced Query (LightRAG)
```bash
POST /query/advanced
Content-Type: application/json

{
  "query": "Find relationships between the main concepts",
  "query_type": "text",
  "mode": "hybrid"
}
```

#### Semantic Search
```bash
POST /query/semantic-search
Content-Type: application/json

{
  "query": "similar documents",
  "limit": 10,
  "entity_type": "document",
  "threshold": 0.7
}
```

#### Hybrid Search
```bash
POST /query/hybrid-search
Content-Type: application/json

{
  "query": "related concepts",
  "vector_weight": 0.7,
  "graph_weight": 0.3,
  "limit": 10
}
```

#### Multi-hop Traversal
```bash
POST /query/multi-hop
Content-Type: application/json

{
  "start_entity": "main_concept",
  "max_hops": 3,
  "relationship_types": ["contains", "references"]
}
```

## 🧪 Testing the API

### Upload a Test Document

1. **Prepare a test document** (PDF, DOCX, XLSX, PPTX, or image)
2. **Upload it via the API**:
   ```bash
   curl -X POST "http://localhost:8000/ingest/upload" \
        -F "file=@your_document.pdf" \
        -F "enable_images=true" \
        -F "enable_tables=true" \
        -F "enable_equations=true"
   ```

3. **Monitor processing**:
   ```bash
   curl http://localhost:8000/ingest/status/{task_id}
   ```

4. **Query the processed document**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{
          "query": "What are the main findings?",
          "query_type": "text",
          "mode": "hybrid"
        }'
   ```

### Python Test Script

```python
import asyncio
import aiohttp
import json

async def test_rag_api():
    # Test data
    test_document = "path/to/your/document.pdf"

    # 1. Upload document
    async with aiohttp.ClientSession() as session:
        with open(test_document, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=test_document)

            async with session.post("http://localhost:8000/ingest/upload", data=data) as resp:
                result = await resp.json()
                task_id = result['task_id']
                print(f"Upload result: {result}")

    # 2. Wait for processing
    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:8000/ingest/status/{task_id}") as resp:
                status = await resp.json()
                print(f"Processing status: {status['status']}")

                if status['status'] == 'completed':
                    break
                elif status['status'] == 'failed':
                    print(f"Processing failed: {status.get('error', 'Unknown error')}")
                    return

                await asyncio.sleep(5)

    # 3. Query the processed document
    query_data = {
        "query": "What are the key findings in this document?",
        "query_type": "text",
        "mode": "hybrid"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/query",
                              json=query_data) as resp:
            result = await resp.json()
            print(f"Query result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_rag_api())
```

## ⚙️ Configuration

### Environment Variables

The application uses the following environment variables (see `env.example`):

#### Required
- `OPENAI_API_KEY`: Your OpenAI API key

#### Optional
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `OPENAI_LLM_MODEL`: LLM model (default: gpt-4o-mini)
- `OPENAI_VISION_MODEL`: Vision model (default: gpt-4o-mini)
- `PARSER`: Document parser (docling or mineru, default: docling)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 100)
- `LIGHTRAG_ENABLED`: Enable LightRAG features (default: true)

### Storage Directories

The application creates and uses these directories:
- `./uploads/`: Temporary file uploads
- `./rag_storage/`: Persistent RAG data (vectors, graphs, metadata)
- `./input/`: Input documents (optional)
- `./output/`: Processing outputs (optional)

## 🐳 Docker Commands

### Build and Run
```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f rag-api

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build --force-recreate
```

### Development with Docker
```bash
# Mount current directory for live reloading
docker run -p 8000:8000 \
  -v $(pwd):/app \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/rag_storage:/app/rag_storage \
  --env-file .env \
  your-image-name
```

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Verify your API key is correct and has sufficient credits
   - Check if the API key has access to the required models

2. **Document Processing Failures**
   - Ensure LibreOffice is installed (for Office documents)
   - Check file format compatibility
   - Verify file size limits

3. **Memory Issues**
   - Increase Docker memory limits
   - Reduce `MAX_WORKERS` in configuration
   - Use smaller batch sizes for processing

4. **Storage Issues**
   - Ensure write permissions on storage directories
   - Check available disk space
   - Clear `rag_storage/` directory if corrupted

### Health Check

Monitor the API health:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "active_tasks": 0,
  "lightrag_enabled": true,
  "config": {
    "max_file_size": 100,
    "parser": "docling",
    "embedding_model": "text-embedding-3-small"
  }
}
```

### Logs

View application logs:
```bash
# Docker Compose
docker-compose logs -f rag-api

# Manual installation
# Logs are printed to console when running uvicorn
```

## 📊 Performance Tips

1. **For Large Documents**
   - Use chunked processing (default behavior)
   - Increase `MAX_WORKERS` for parallel processing
   - Monitor memory usage

2. **For High Query Volume**
   - Enable caching in configuration
   - Consider using external vector databases (Qdrant)
   - Implement query result caching

3. **Storage Optimization**
   - Regular cleanup of old documents
   - Use efficient storage backends
   - Monitor storage growth

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Check the application logs
4. Ensure all dependencies are properly installed

For additional help, please open an issue in the repository.
