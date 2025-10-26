from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union

# ----- Normalized content items (mirrors RAG-Anything)
class TextItem(BaseModel):
    type: Literal["text"] = "text"
    text: str
    page_idx: Optional[int] = None
    headers: Optional[List[str]] = None

class ImageItem(BaseModel):
    type: Literal["image"] = "image"
    img_path: str
    image_caption: Optional[List[str]] = None
    image_footnote: Optional[List[str]] = None
    page_idx: Optional[int] = None

class TableItem(BaseModel):
    type: Literal["table"] = "table"
    table_body: str                     # markdown / csv-like
    img_path: Optional[str] = None      # snapshot if available
    table_caption: Optional[List[str]] = None
    table_footnote: Optional[List[str]] = None
    page_idx: Optional[int] = None

class EquationItem(BaseModel):
    type: Literal["equation"] = "equation"
    text: str                 # LaTeX or plain
    text_format: Optional[str] = None
    page_idx: Optional[int] = None

ContentItem = TextItem | ImageItem | TableItem | EquationItem

# ----- Requests / Responses
class IngestOptions(BaseModel):
    parse_method: Optional[str] = Field(default=None)
    lang: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    enable_table: Optional[bool] = True
    enable_formula: Optional[bool] = True
    backend: Optional[str] = None
    source: Optional[str] = None
    excel_native_summary: Optional[bool] = True   # Path B for large Excel

class IngestResponse(BaseModel):
    doc_id: str
    file_name: str
    total_blocks: int
    by_type: Dict[str, int]
    notes: Optional[str] = None
    is_duplicate: bool = False
    duplicate_doc_id: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    mode: Literal["text","multimodal","vlm_enhanced"] = "text"
    k: int = 8
    filters: Optional[Dict[str, Any]] = None
    doc_ids: Optional[List[str]] = None
    include_context: bool = True

class QueryHit(BaseModel):
    chunk_id: str
    score: float
    doc_id: str
    type: str
    page_idx: Optional[int] = None
    text_preview: str

class QueryResponse(BaseModel):
    answer: str
    hits: List[QueryHit] = []
    used_mode: str

class StatusResponse(BaseModel):
    docs_indexed: int
    chunks_indexed: int

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_size: int
    total_blocks: int
    by_type: Dict[str, int]
    processed_at: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int
