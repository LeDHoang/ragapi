from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"

class TextContent(BaseModel):
    type: str = ContentType.TEXT
    text: str
    text_level: int = 0  # 0=paragraph, 1=h1, 2=h2, etc.
    page_idx: int

class ImageContent(BaseModel):
    type: str = ContentType.IMAGE
    img_path: str
    image_caption: Optional[List[str]] = None
    image_footnote: Optional[List[str]] = None
    page_idx: int

class TableContent(BaseModel):
    type: str = ContentType.TABLE
    table_body: str
    table_caption: Optional[List[str]] = None
    table_footnote: Optional[List[str]] = None
    page_idx: int

class EquationContent(BaseModel):
    type: str = ContentType.EQUATION
    latex: str
    text: Optional[str] = None
    page_idx: int

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: float = 0.0
    error: Optional[str] = None
    doc_id: Optional[str] = None
    chunks_created: Optional[int] = None
    entities_found: Optional[int] = None

class DocumentMetadata(BaseModel):
    doc_id: str
    file_path: str
    file_type: str
    total_pages: int
    processed_at: float
    chunks_count: int
    entities_count: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    query_type: str = "text"  # "text", "multimodal", "vlm_enhanced"
    multimodal_content: Optional[List[Dict[str, Any]]] = None
    mode: str = "hybrid"  # "local", "global", "hybrid", "naive"

class QueryResponse(BaseModel):
    result: str
    query_type: str
    processing_time: float
    entities_found: List[str] = []
    multimodal_context: List[str] = []

class EntityNode(BaseModel):
    entity_id: str
    entity_type: str
    name: str
    description: str
    source_id: str
    file_path: str
    created_at: float

class EntityRelation(BaseModel):
    src_id: str
    tgt_id: str
    relation_type: str
    description: str
    keywords: str
    source_id: str
    weight: float
    file_path: str