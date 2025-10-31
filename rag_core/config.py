from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env into process environment early, so os.getenv picks it up regardless of CWD
load_dotenv()  # Uses default .env in current or parent directories

class RAGConfig(BaseSettings):
    # Base paths
    WORKING_DIR: str = "./rag_storage"
    UPLOAD_DIR: str = "./uploads"
    
    # Parser configuration
    PARSER: str = "mineru"  # Primary parser for Office documents, PDFs, images
    PARSE_METHOD: str = "auto"  # "auto", "ocr", "txt"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-v2"
    
    # OpenAI Configuration (read directly from environment)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Model Configuration (with environment variable mapping)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_EMBEDDING_MODEL: Optional[str] = os.getenv("OPENAI_EMBEDDING_MODEL")  # For .env compatibility
    EMBEDDING_DIM: int = 1536
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_LLM_MODEL: Optional[str] = os.getenv("OPENAI_LLM_MODEL")  # For .env compatibility
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o-mini")
    OPENAI_VISION_MODEL: Optional[str] = os.getenv("OPENAI_VISION_MODEL")  # For .env compatibility
    LIGHTRAG_ENABLED: bool = True  # Set to False to disable LightRAG
    LIGHTRAG_WORKING_DIR: Optional[str] = None
    LIGHTRAG_KV_STORAGE: str = "JsonKVStorage"
    LIGHTRAG_VECTOR_STORAGE: str = "NanoVectorDBStorage"
    LIGHTRAG_GRAPH_STORAGE: str = "NetworkXStorage"
    LIGHTRAG_DOC_STATUS_STORAGE: str = "JsonDocStatusStorage"
    
    # Processing Configuration
    MAX_FILE_SIZE_MB: int = 100
    MAX_WORKERS: int = 4
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Database Configuration
    VECTOR_DB: str = "local://vectors"  # Use local storage instead of Qdrant
    GRAPH_DB: str = "neo4j://localhost:7687"
    GRAPH_DB_USER: str = ""  # No authentication for testing
    GRAPH_DB_PASSWORD: str = ""
    CACHE_DB: str = "redis://localhost:6379"
    
    # Content Processing
    ENABLE_IMAGES: bool = True
    ENABLE_TABLES: bool = True
    ENABLE_EQUATIONS: bool = True

    # LibreOffice Online conversion (Collabora CODE)
    LOOL_ENABLED: bool = True
    LOOL_BASE_URL: str = os.getenv("LOOL_BASE_URL", "http://localhost:9980")
    LOOL_ENDPOINT: str = os.getenv("LOOL_ENDPOINT", "/lool/convert-to/pdf")
    LOOL_TIMEOUT: int = int(os.getenv("LOOL_TIMEOUT", "600"))
    LOOL_RETRY_ATTEMPTS: int = int(os.getenv("LOOL_RETRY_ATTEMPTS", "3"))
    LOOL_RETRY_DELAY: float = float(os.getenv("LOOL_RETRY_DELAY", "5.0"))
    LOOL_FALLBACK_TO_LOCAL: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from environment

    def get_working_dir(self) -> Path:
        return Path(self.WORKING_DIR).absolute()

    def get_lightrag_working_dir(self) -> Path:
        if self.LIGHTRAG_WORKING_DIR:
            return Path(self.LIGHTRAG_WORKING_DIR).absolute()
        return self.get_working_dir()
    
    def get_upload_dir(self) -> Path:
        return Path(self.UPLOAD_DIR).absolute()
    
    def get_parser_config(self) -> Dict[str, Any]:
        return {
            "parser": self.PARSER,
            "method": self.PARSE_METHOD,
            "enable_images": self.ENABLE_IMAGES,
            "enable_tables": self.ENABLE_TABLES,
            "enable_equations": self.ENABLE_EQUATIONS
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        # Use environment-specific values if available, otherwise use defaults
        embedding_model = self.OPENAI_EMBEDDING_MODEL or self.EMBEDDING_MODEL
        llm_model = self.OPENAI_LLM_MODEL or self.LLM_MODEL
        vision_model = self.OPENAI_VISION_MODEL or self.VISION_MODEL
        
        return {
            "embedding_model": embedding_model,
            "embedding_dim": self.EMBEDDING_DIM,
            "llm_model": llm_model,
            "vision_model": vision_model
        }

    def get_lightrag_config(self) -> Dict[str, Any]:
        return {
            "kv_storage": self.LIGHTRAG_KV_STORAGE,
            "vector_storage": self.LIGHTRAG_VECTOR_STORAGE,
            "graph_storage": self.LIGHTRAG_GRAPH_STORAGE,
            "doc_status_storage": self.LIGHTRAG_DOC_STATUS_STORAGE,
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        return {
            "max_file_size": self.MAX_FILE_SIZE_MB * 1024 * 1024,
            "max_workers": self.MAX_WORKERS,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP
        }

# Create global config instance
config = RAGConfig()

# Ensure directories exist
os.makedirs(config.get_working_dir(), exist_ok=True)
os.makedirs(config.get_upload_dir(), exist_ok=True)
