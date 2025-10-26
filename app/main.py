from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional
from rag_core.schemas import (
    IngestOptions, IngestResponse, QueryRequest, QueryResponse, StatusResponse
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
