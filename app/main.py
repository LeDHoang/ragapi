from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional
import asyncio
from rag_core.schemas import (
    IngestOptions, IngestResponse, QueryRequest, QueryResponse, StatusResponse,
    DocumentListResponse, DocumentInfo
)
from app.deps import get_pipeline, get_config
from rag_core.pipeline import RagPipeline
from rag_core.config import AppConfig

app = FastAPI(title="RAG-Anything Style API", version="0.1.0")

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    opts: IngestOptions = Depends(),
    pipeline: RagPipeline = Depends(get_pipeline),
):
    try:
        return await pipeline.ingest_file(file, opts)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")

@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest = Body(...),
    pipeline: RagPipeline = Depends(get_pipeline),
):
    try:
        result = await pipeline.query(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")

@app.get("/status", response_model=StatusResponse)
async def status(
    pipeline: RagPipeline = Depends(get_pipeline),
    cfg: AppConfig = Depends(get_config),
):
    return await pipeline.status()

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    pipeline: RagPipeline = Depends(get_pipeline),
):
    """List all processed documents"""
    try:
        return await pipeline.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    pipeline: RagPipeline = Depends(get_pipeline),
):
    """Delete a document and its associated data"""
    try:
        success = await pipeline.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

@app.get("/documents/{doc_id}")
async def get_document_info(
    doc_id: str,
    pipeline: RagPipeline = Depends(get_pipeline),
):
    """Get information about a specific document"""
    try:
        # First try to get from document registry
        doc_info = pipeline.stores.documents.get_document_info(doc_id)
        if doc_info:
            return DocumentInfo(
                doc_id=doc_id,
                filename=doc_info["filename"],
                file_size=doc_info["file_size"],
                total_blocks=doc_info["total_blocks"],
                by_type=doc_info["by_type"],
                processed_at=doc_info["processed_at"]
            )
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {e}")
