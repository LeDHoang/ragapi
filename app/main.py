from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from pathlib import Path
import asyncio
import time
import json
import shutil
from typing import List, Optional

from rag_core.config import config
from rag_core.pipeline import RAGPipeline
from rag_core.query import QueryProcessor
from rag_core.advanced_query import AdvancedQueryProcessor
from rag_core.conversion.excel_to_pdf import (
    convert_excel_to_pdf,
    get_latest_debug_artifacts_dir,
    get_latest_debug_combined_pdf,
    needs_conversion,
)
from rag_core.parsers import ParserFactory
from rag_core.schemas import (
    ProcessingStatus,
    DocumentMetadata,
    QueryRequest,
    QueryResponse
)
from pydantic import BaseModel

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

# Log key configuration at startup (masked)
try:
    key = config.OPENAI_API_KEY
    base_url = config.OPENAI_BASE_URL
    masked = None
    if key:
        if len(key) >= 12:
            masked = f"{key[:6]}...{key[-4:]}"
        else:
            masked = "[set]"
    else:
        masked = "[not set]"

    from pathlib import Path as _Path
    env_here = _Path(".env")
    logger.info(
        "[CONFIG] OPENAI_API_KEY=%s OPENAI_BASE_URL=%s .env_present=%s cwd=%s",
        masked,
        base_url,
        env_here.exists(),
        str(_Path.cwd())
    )
except Exception as _e:
    logger.warning("[CONFIG] Failed to log OpenAI config: %s", str(_e))

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
        request_start = time.time()
        logger.info("[INGEST] Upload started: filename=%s", file.filename)
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
        logger.info(
            "[INGEST] File saved: task_id=%s path=%s elapsed=%.3fs",
            task_id,
            str(file_path),
            time.time() - request_start
        )

        # Check if file needs conversion and convert if necessary
        if needs_conversion(str(file_path)):
            logger.info("[INGEST] File needs conversion, converting to PDF: %s", file_path)
            conversion_start = time.time()
            try:
                converted_pdf_path = convert_excel_to_pdf(str(file_path))  # Will save to input/converted
                file_path = Path(converted_pdf_path)  # Update file_path to point to the PDF
                debug_dir = get_latest_debug_artifacts_dir()
                debug_combined = get_latest_debug_combined_pdf()
                conversion_elapsed = time.time() - conversion_start
                logger.info(
                    "[INGEST] Conversion completed: original=%s pdf=%s conversion_time=%.2fs total_elapsed=%.2fs",
                    str(upload_dir / file.filename),
                    str(file_path),
                    conversion_elapsed,
                    time.time() - request_start
                )
                if debug_dir:
                    logger.info("[INGEST] Debug artifacts stored at %s", debug_dir)
                if debug_combined:
                    logger.info("[INGEST] Debug combined PDF at %s", debug_combined)
            except Exception as conv_error:
                logger.error("[INGEST] Conversion failed after %.2fs for %s: %s",
                           time.time() - conversion_start, file_path, str(conv_error))
                raise HTTPException(
                    status_code=500,
                    detail=f"Document conversion failed: {str(conv_error)}"
                )

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
        logger.info(
            "[INGEST] Background task queued: task_id=%s elapsed=%.3fs",
            task_id,
            time.time() - request_start
        )
        
        # Initialize task status
        processing_tasks[task_id] = {
            "status": "processing",
            "file_path": str(file_path),
            "start_time": time.time()
        }
        logger.info(
            "[INGEST] Tracking initialized: task_id=%s total_elapsed=%.3fs",
            task_id,
            time.time() - request_start
        )
        
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

@app.post("/convert")
async def convert_document(file: UploadFile = File(...)):
    """Convert Excel/Office document to PDF and return the PDF file"""

    try:
        request_start = time.time()
        logger.info("[CONVERT] Conversion started: filename=%s", file.filename)

        # Validate file size
        if file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {config.MAX_FILE_SIZE_MB}MB)"
            )

        # Check if file needs conversion
        file_ext = Path(file.filename).suffix.lower()
        if not needs_conversion(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} does not need conversion. Supported: Excel (.xlsx, .xls)"
            )

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Create upload directory for original file
        upload_dir = config.get_upload_dir() / task_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save original file
        original_file_path = upload_dir / file.filename
        content = await file.read()
        with open(original_file_path, "wb") as f:
            f.write(content)

        # Convert to PDF (will save to input/converted directory)
        conversion_start = time.time()
        try:
            logger.info("[CONVERT] Starting conversion for task_id=%s: %s", task_id, original_file_path)
            converted_pdf_path = convert_excel_to_pdf(str(original_file_path))
            conversion_elapsed = time.time() - conversion_start
            logger.info(
                "[CONVERT] Conversion completed: task_id=%s original=%s pdf=%s conversion_time=%.2fs total_elapsed=%.2fs",
                task_id,
                str(original_file_path),
                converted_pdf_path,
                conversion_elapsed,
                time.time() - request_start
            )

            return {
                "task_id": task_id,
                "status": "converted",
                "original_file": str(original_file_path),
                "converted_pdf": converted_pdf_path,
                "debug_artifacts_dir": get_latest_debug_artifacts_dir(),
                "debug_combined_pdf": get_latest_debug_combined_pdf(),
                "message": "Document conversion successful"
            }

        except Exception as conv_error:
            logger.error(f"Conversion failed for {original_file_path}: {str(conv_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Document conversion failed: {str(conv_error)}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document conversion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document conversion failed: {str(e)}"
        )

@app.post("/process")
async def process_document_preview(
    file: UploadFile = File(...),
    enable_images: bool = True,
    enable_tables: bool = True,
    enable_equations: bool = True,
    parser: str = None
):
    """Run the preprocessing pipeline (conversion + parsing) and return intermediate artifacts."""

    upload_dir = None
    original_file_path: Optional[Path] = None
    working_file_path: Optional[Path] = None
    debug_dir: Optional[str] = None
    debug_combined: Optional[str] = None

    try:
        request_start = time.time()
        logger.info("[PROCESS] Upload started: filename=%s", file.filename)

        if file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {config.MAX_FILE_SIZE_MB}MB)"
            )

        process_id = str(uuid.uuid4())
        upload_dir = config.get_upload_dir() / process_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        original_file_path = upload_dir / file.filename
        content = await file.read()
        with open(original_file_path, "wb") as f:
            f.write(content)
        logger.info("[PROCESS] File saved: %s", original_file_path)

        working_file_path = original_file_path

        if needs_conversion(str(original_file_path)):
            logger.info("[PROCESS] File needs conversion, converting to PDF: %s", original_file_path)
            conversion_start = time.time()
            try:
                converted_pdf_path = convert_excel_to_pdf(str(original_file_path))
                working_file_path = Path(converted_pdf_path)
                logger.info(
                    "[PROCESS] Conversion completed: pdf=%s conversion_time=%.2fs",
                    working_file_path,
                    time.time() - conversion_start
                )
                debug_dir = get_latest_debug_artifacts_dir()
                debug_combined = get_latest_debug_combined_pdf()
            except Exception as conv_error:
                logger.error("[PROCESS] Conversion failed: %s", str(conv_error))
                raise HTTPException(
                    status_code=500,
                    detail=f"Document conversion failed: {str(conv_error)}"
                )

        ingest_summary = {
            "parser_used": None,
            "storage_issues": [],
            "errors": {},
            "warnings": {}
        }

        # Parse document to obtain structured content
        content_list = await ParserFactory.parse_document(
            str(working_file_path),
            parser_type=parser,
            ingest_summary=ingest_summary
        )

        if not content_list:
            raise HTTPException(
                status_code=422,
                detail="Parsing produced no content for the supplied document"
            )

        # Apply optional modality filters
        def _keep_item(item: dict) -> bool:
            raw_type = item.get("type", "text")
            kind = raw_type.value if hasattr(raw_type, "value") else str(raw_type).lower()
            if kind == "image" and not enable_images:
                return False
            if kind == "table" and not enable_tables:
                return False
            if kind == "equation" and not enable_equations:
                return False
            return True

        filtered_content_list = [item for item in content_list if _keep_item(item)]
        if len(filtered_content_list) != len(content_list):
            logger.info(
                "[PROCESS] Filtered content items: kept %s of %s after modality toggles",
                len(filtered_content_list),
                len(content_list)
            )

        # Separate content into text and multimodal items
        full_text, multimodal_items, summary = await pipeline.content_separator.process_document_content(
            filtered_content_list, process_id
        )

        doc_stem = Path(working_file_path).stem
        output_root = config.get_working_dir() / "output" / doc_stem
        auto_dir = output_root / "auto"
        auto_dir.mkdir(parents=True, exist_ok=True)

        # Persist intermediate artifacts
        content_path = auto_dir / f"{doc_stem}_content_list.json"
        with open(content_path, "w", encoding="utf-8") as f:
            json.dump(filtered_content_list, f, ensure_ascii=False, indent=2)

        middle_path = auto_dir / f"{doc_stem}_middle.json"
        with open(middle_path, "w", encoding="utf-8") as f:
            json.dump(summary.get("structure", {}), f, ensure_ascii=False, indent=2)

        model_data = {
            "doc_id": doc_stem,
            "full_text": full_text,
            "multimodal_items": multimodal_items,
            "summary": summary,
            "content_list": filtered_content_list
        }
        model_path = auto_dir / f"{doc_stem}_model.json"
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        markdown_path = auto_dir / f"{doc_stem}.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(full_text or "")

        # Copy converted/original files for reference
        try:
            if working_file_path and working_file_path.exists():
                shutil.copy2(working_file_path, auto_dir / f"{doc_stem}_origin{working_file_path.suffix}")
        except Exception as copy_error:
            logger.warning("[PROCESS] Failed to copy converted file: %s", copy_error)

        try:
            if (
                original_file_path
                and original_file_path.exists()
                and working_file_path
                and original_file_path != working_file_path
            ):
                shutil.copy2(original_file_path, auto_dir / original_file_path.name)
        except Exception as copy_error:
            logger.warning("[PROCESS] Failed to copy original upload: %s", copy_error)

        logger.info(
            "[PROCESS] Processing completed in %.2fs. Outputs stored at %s",
            time.time() - request_start,
            auto_dir
        )

        if debug_dir:
            logger.info("[PROCESS] Debug artifacts stored at %s", debug_dir)
        if debug_combined:
            logger.info("[PROCESS] Debug combined PDF at %s", debug_combined)

        return {
            "status": "processed",
            "task_id": process_id,
            "output_dir": str(auto_dir),
            "files": {
                "content_list": str(content_path),
                "middle": str(middle_path),
                "model": str(model_path),
                "markdown": str(markdown_path)
            },
            "debug_artifacts_dir": debug_dir,
            "debug_combined_pdf": debug_combined,
            "parser_used": ingest_summary.get("parser_used"),
            "warnings": ingest_summary.get("warnings", {}),
            "errors": ingest_summary.get("errors", {})
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PROCESS] Document processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )
    finally:
        try:
            if original_file_path and original_file_path.exists():
                original_file_path.unlink(missing_ok=True)
            if (
                working_file_path
                and working_file_path.exists()
                and original_file_path
                and working_file_path != original_file_path
            ):
                working_file_path.unlink(missing_ok=True)
            if upload_dir and upload_dir.exists():
                try:
                    next(upload_dir.iterdir())
                except StopIteration:
                    upload_dir.rmdir()
        except Exception as cleanup_error:
            logger.debug("[PROCESS] Cleanup skipped: %s", cleanup_error)

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
        bg_start = time.time()
        logger.info("[INGEST] Processing started: task_id=%s file=%s", task_id, file_path)
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
        started_at = processing_tasks[task_id].get("start_time", bg_start)
        logger.info(
            "[INGEST] Processing completed: task_id=%s doc_id=%s chunks=%s entities=%s elapsed=%.3fs total_elapsed=%.3fs",
            task_id,
            status.doc_id,
            status.chunks_created,
            status.entities_found,
            time.time() - bg_start,
            time.time() - started_at
        )
        
        # Clean up upload file
        Path(file_path).unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        processing_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": time.time()
        })
        started_at = processing_tasks[task_id].get("start_time", None)
        if started_at:
            logger.info(
                "[INGEST] Processing failed: task_id=%s elapsed=%.3fs",
                task_id,
                time.time() - started_at
            )
        
        # Clean up on error
        Path(file_path).unlink(missing_ok=True)
