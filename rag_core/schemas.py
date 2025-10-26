from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

class IngestOptions(BaseModel):
    """Options for document ingestion"""
    enable_images: bool = True
    enable_tables: bool = True
    enable_equations: bool = True
    export_layout_overlay: bool = False
    parser: str = "auto"  # "auto", "mineru", "docling"
    parse_method: str = "auto"  # "auto", "ocr", "txt"
    
    # Additional fields that pipeline expects
    lang: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    enable_formula: bool = True  # Alias for enable_equations
    enable_table: bool = True    # Alias for enable_tables
    enable_image: bool = True    # Alias for enable_images
    backend: Optional[str] = None
    source: Optional[str] = None
    excel_native_summary: bool = False

class IngestResponse(BaseModel):
    """Response from document ingestion"""
    doc_id: str
    file_name: str  # Changed from filename to file_name to match pipeline
    file_size: int
    total_blocks: int
    by_type: Dict[str, int]
    processed_at: datetime
    success: bool = True
    message: Optional[str] = None
    overlay_path: Optional[str] = None
    notes: Optional[str] = None  # Add notes field that pipeline uses
    is_duplicate: bool = False  # Add duplicate detection fields
    duplicate_doc_id: Optional[str] = None

class QueryRequest(BaseModel):
    """Request for querying the RAG system"""
    query: str
    mode: Literal["text", "multimodal", "auto"] = "auto"
    k: int = 10  # Number of search results to retrieve
    max_hits: int = 10
    include_images: bool = True
    include_tables: bool = True
    include_equations: bool = True

class QueryResponse(BaseModel):
    """Response from querying the RAG system"""
    answer: str
    hits: List[Dict[str, Any]]
    used_mode: str
    processing_time: Optional[float] = None

class StatusResponse(BaseModel):
    """System status response"""
    docs_indexed: int
    chunks_indexed: int
    status: str = "healthy"

class DocumentInfo(BaseModel):
    """Information about a specific document"""
    doc_id: str
    filename: str
    file_size: int
    total_blocks: int
    by_type: Dict[str, int]
    processed_at: datetime

class DocumentListResponse(BaseModel):
    """Response for listing documents"""
    documents: List[DocumentInfo]
    total: int
