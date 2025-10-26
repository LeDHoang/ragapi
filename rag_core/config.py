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

    # Multimodal processing options
    context_window: int = field(default=int(os.getenv("CONTEXT_WINDOW", "1")))
    context_mode: str   = field(default=os.getenv("CONTEXT_MODE","page"))
    max_context_tokens: int = field(default=int(os.getenv("MAX_CONTEXT_TOKENS", "2000")))
    include_headers: bool   = field(default=os.getenv("INCLUDE_HEADERS","true").lower()=="true")
    include_captions: bool  = field(default=os.getenv("INCLUDE_CAPTIONS","true").lower()=="true")
    context_filter_content_types: List[str] = field(
        default_factory=lambda: os.getenv("CONTEXT_FILTER_CONTENT_TYPES","text").split(",")
    )

    # Large document processing
    parse_chunk_size_pages: int = field(default=int(os.getenv("PARSE_CHUNK_SIZE_PAGES", "0")))
    table_body_max_chars: int = field(default=int(os.getenv("TABLE_BODY_MAX_CHARS", "5000")))

    # Layout overlay options
    export_layout_overlay: bool = field(default=os.getenv("EXPORT_LAYOUT_OVERLAY","false").lower()=="true")
    overlay_dpi: int = field(default=int(os.getenv("OVERLAY_DPI", "144")))
    overlay_dir: str = field(default=os.getenv("OVERLAY_DIR", "./output/layout_overlays"))

    # LLM Provider selection - auto-detect based on available credentials
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER",
        "openai" if os.getenv("OPENAI_API_KEY") else "bedrock"))  # bedrock|openai

    # Bedrock model configuration
    bedrock_region: str = field(default_factory=lambda: os.getenv("BEDROCK_REGION", "us-east-1"))
    bedrock_llm_model_id: str = field(default_factory=lambda: os.getenv("BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"))
    bedrock_vision_model_id: str = field(default_factory=lambda: os.getenv("BEDROCK_VISION_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"))
    bedrock_embedding_model_id: str = field(default_factory=lambda: os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"))
    bedrock_embedding_dim: int = field(default_factory=lambda: int(os.getenv("BEDROCK_EMBEDDING_DIM", "1024")))

    # OpenAI model configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    openai_vision_model: str = field(default_factory=lambda: os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    openai_embedding_dim: int = field(default_factory=lambda: int(os.getenv("OPENAI_EMBEDDING_DIM", "3072")))

    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "8")))

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig()
