from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from pathlib import Path
import asyncio
import time

from rag_core.config import config
from rag_core.pipeline import RAGPipeline
from rag_core.query import QueryProcessor
from rag_core.schemas import (
    ProcessingStatus,
    DocumentMetadata,
    QueryRequest,
    QueryResponse
)

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
query_processor = QueryProcessor()

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
        return await query_processor.process_query(request)
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
        "config": {
            "max_file_size": config.MAX_FILE_SIZE_MB,
            "parser": config.PARSER,
            "embedding_model": config.EMBEDDING_MODEL
        }
    }

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