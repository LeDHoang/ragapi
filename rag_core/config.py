import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import List

load_dotenv()

@dataclass
class AppConfig:
    working_dir: str = field(default=os.getenv("WORKING_DIR", "./rag_storage"))
    output_dir: str  = field(default=os.getenv("OUTPUT_DIR", "./output"))
    parser: str      = field(default=os.getenv("PARSER", "docling"))   # docling|mineru
    parse_method: str= field(default=os.getenv("PARSE_METHOD", "auto"))

    enable_image_processing: bool = field(default=os.getenv("ENABLE_IMAGE_PROCESSING","true").lower()=="true")
    enable_table_processing: bool = field(default=os.getenv("ENABLE_TABLE_PROCESSING","true").lower()=="true")
    enable_equation_processing: bool = field(default=os.getenv("ENABLE_EQUATION_PROCESSING","true").lower()=="true")

    context_window: int = field(default=int(os.getenv("CONTEXT_WINDOW", "1")))
    context_mode: str   = field(default=os.getenv("CONTEXT_MODE","page"))
    max_context_tokens: int = field(default=int(os.getenv("MAX_CONTEXT_TOKENS", "2000")))
    include_headers: bool   = field(default=os.getenv("INCLUDE_HEADERS","true").lower()=="true")
    include_captions: bool  = field(default=os.getenv("INCLUDE_CAPTIONS","true").lower()=="true")
    context_filter_content_types: List[str] = field(
        default_factory=lambda: os.getenv("CONTEXT_FILTER_CONTENT_TYPES","text").split(",")
    )

    top_k: int = field(default=int(os.getenv("TOP_K", "8")))

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig()
