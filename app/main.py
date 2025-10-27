from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from pathlib import Path
import asyncio
import time
from typing import List

from rag_core.config import config
from rag_core.pipeline import RAGPipeline
from rag_core.query import QueryProcessor
from rag_core.advanced_query import AdvancedQueryProcessor
from rag_core.schemas import (
    ProcessingStatus,
    DocumentMetadata,
    QueryRequest,
    QueryResponse
)
from pydantic import BaseModel
from typing import Optional

class SemanticSearchRequest(BaseModel):
    query: str
    limit: int = 10
    entity_type: Optional[str] = None
    threshold: float = 0.7

class HybridSearchRequest(BaseModel):
    query: str
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    limit: int = 10

class MultiHopRequest(BaseModel):
    start_entity: str
    max_hops: int = 3
    relationship_types: Optional[List[str]] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG-Anything API",
    description="Multimodal RAG API for document processing and querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
pipeline = RAGPipeline()
legacy_query_processor = QueryProcessor()
advanced_query_processor = AdvancedQueryProcessor(pipeline.lightrag)

# Task management
processing_tasks = {}

@app.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_images: bool = True,
    enable_tables: bool = True,
    enable_equations: bool = True,
    parser: str = None,
    export_layout_overlay: bool = False
):
    """Ingest document for processing (alias for /ingest/upload)"""
    return await upload_document(
        background_tasks=background_tasks,
        file=file,
        enable_images=enable_images,
        enable_tables=enable_tables,
        enable_equations=enable_equations,
        parser=parser
    )

@app.post("/ingest/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_images: bool = True,
    enable_tables: bool = True,
    enable_equations: bool = True,
    parser: str = None
):
    """Upload document for processing"""
    
    try:
        # Validate file size
        if file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {config.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create upload directory
        upload_dir = config.get_upload_dir() / task_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            task_id=task_id,
            file_path=str(file_path),
            parser_type=parser,
            config_overrides={
                "enable_images": enable_images,
                "enable_tables": enable_tables,
                "enable_equations": enable_equations
            }
        )
        
        # Initialize task status
        processing_tasks[task_id] = {
            "status": "processing",
            "file_path": str(file_path),
            "start_time": time.time()
        }
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Document upload successful, processing started"
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )

@app.get("/ingest/status/{task_id}")
async def get_processing_status(task_id: str):
    """Get document processing status"""
    
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    task_info = processing_tasks[task_id].copy()
    
    # Add processing time
    task_info["processing_time"] = time.time() - task_info["start_time"]
    
    # Clean up completed tasks after 1 hour
    if (task_info["status"] in ["completed", "failed"] and
        task_info["processing_time"] > 3600):
        del processing_tasks[task_id]
    
    return task_info

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """Query the knowledge base"""

    try:
        # Use advanced query processor if LightRAG is available and enabled
        if pipeline.lightrag and config.LIGHTRAG_ENABLED:
            return await advanced_query_processor.process_query_lightrag(request)
        else:
            # Fallback to legacy query processor
            return await legacy_query_processor.process_query(request)
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    
    try:
        documents = pipeline.doc_registry.list_documents()
        return {
            "documents": documents,
            "total": len(documents)
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document details"""
    
    try:
        document = pipeline.doc_registry.get_document(doc_id)
        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document: {str(e)}"
        )

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document and associated data"""
    
    try:
        # Get document info
        document = pipeline.doc_registry.get_document(doc_id)
        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        # Delete document file
        file_path = Path(document["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Delete from registry
        pipeline.doc_registry.remove_document(doc_id)
        
        # Delete chunks
        chunks = pipeline.chunk_manager.get_chunks_by_doc(doc_id)
        for chunk in chunks:
            chunk_file = pipeline.chunk_manager.chunks_dir / f"{chunk['id']}.txt"
            if chunk_file.exists():
                chunk_file.unlink()
        
        return {
            "message": "Document deleted successfully",
            "doc_id": doc_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""

    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_tasks": len(processing_tasks),
        "lightrag_enabled": config.LIGHTRAG_ENABLED and pipeline.lightrag is not None,
        "config": {
            "max_file_size": config.MAX_FILE_SIZE_MB,
            "parser": config.PARSER,
            "embedding_model": config.EMBEDDING_MODEL
        }
    }

# Advanced LightRAG endpoints
@app.post("/query/advanced")
async def advanced_query(request: QueryRequest):
    """Advanced query using LightRAG's enhanced capabilities"""

    try:
        if not pipeline.lightrag or not config.LIGHTRAG_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="LightRAG not available. Please ensure LightRAG is enabled and initialized."
            )

        return await advanced_query_processor.process_query_lightrag(request)
    except Exception as e:
        logger.error(f"Advanced query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced query failed: {str(e)}"
        )

@app.post("/query/semantic-search")
async def semantic_similarity_search(request: SemanticSearchRequest):
    """Semantic similarity search using LightRAG's vector databases"""

    try:
        if not pipeline.lightrag or not config.LIGHTRAG_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="LightRAG not available. Please ensure LightRAG is enabled and initialized."
            )

        results = await advanced_query_processor.semantic_similarity_search(
            query=request.query,
            limit=request.limit,
            entity_type=request.entity_type,
            threshold=request.threshold
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "lightrag_enabled": True
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )

@app.post("/query/hybrid-search")
async def hybrid_search(request: HybridSearchRequest):
    """Hybrid search combining vector and graph search using LightRAG"""

    try:
        if not pipeline.lightrag or not config.LIGHTRAG_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="LightRAG not available. Please ensure LightRAG is enabled and initialized."
            )

        results = await advanced_query_processor.hybrid_search(
            query=request.query,
            vector_weight=request.vector_weight,
            graph_weight=request.graph_weight,
            limit=request.limit
        )

        return {
            "query": request.query,
            "results": results,
            "lightrag_enabled": True
        }
    except Exception as e:
        logger.error(f"Hybrid search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid search failed: {str(e)}"
        )

@app.post("/query/multi-hop")
async def multi_hop_traversal(request: MultiHopRequest):
    """Multi-hop graph traversal using LightRAG"""

    try:
        if not pipeline.lightrag or not config.LIGHTRAG_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="LightRAG not available. Please ensure LightRAG is enabled and initialized."
            )

        results = await advanced_query_processor.multi_hop_traversal(
            start_entity=request.start_entity,
            max_hops=request.max_hops,
            relationship_types=request.relationship_types
        )

        return {
            "start_entity": request.start_entity,
            "results": results,
            "lightrag_enabled": True
        }
    except Exception as e:
        logger.error(f"Multi-hop traversal failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-hop traversal failed: {str(e)}"
        )

@app.get("/query/entity-relationships/{entity_id}")
async def get_entity_relationships(
    entity_id: str,
    max_depth: int = 2,
    relationship_types: str = None  # Comma-separated string
):
    """Get entity relationships using LightRAG's graph traversal"""

    try:
        if not pipeline.lightrag or not config.LIGHTRAG_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="LightRAG not available. Please ensure LightRAG is enabled and initialized."
            )

        # Parse relationship types if provided
        rel_types = None
        if relationship_types:
            rel_types = [rt.strip() for rt in relationship_types.split(",")]

        results = await advanced_query_processor.get_entity_relationships(
            entity_id=entity_id,
            max_depth=max_depth,
            relationship_types=rel_types
        )

        return {
            "entity_id": entity_id,
            "results": results,
            "lightrag_enabled": True
        }
    except Exception as e:
        logger.error(f"Entity relationships query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Entity relationships query failed: {str(e)}"
        )

async def process_document_background(
    task_id: str,
    file_path: str,
    parser_type: str = None,
    config_overrides: dict = None
):
    """Background document processing"""
    
    try:
        # Update task status
        processing_tasks[task_id].update({
            "status": "processing",
            "progress": 0.0
        })
        
        # Process document
        status = await pipeline.process_document(
            file_path=file_path,
            task_id=task_id,
            parser_type=parser_type
        )
        
        # Update task status
        processing_tasks[task_id].update({
            "status": status.status,
            "progress": status.progress,
            "doc_id": status.doc_id,
            "chunks_created": status.chunks_created,
            "entities_found": status.entities_found,
            "completed_at": time.time()
        })
        
        # Clean up upload file
        Path(file_path).unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        processing_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": time.time()
        })
        
        # Clean up on error
        Path(file_path).unlink(missing_ok=True)