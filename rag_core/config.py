from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os
from pathlib import Path

class RAGConfig(BaseSettings):
    # Base paths
    WORKING_DIR: str = "./rag_storage"
    UPLOAD_DIR: str = "./uploads"
    
    # Parser configuration
    PARSER: str = "docling"  # or "mineru"
    PARSE_METHOD: str = "auto"  # "auto", "ocr", "txt"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-v2"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    
    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIM: int = 3072
    LLM_MODEL: str = "gpt-4o-mini"
    VISION_MODEL: str = "gpt-4o-mini"
    
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
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from environment

    def get_working_dir(self) -> Path:
        return Path(self.WORKING_DIR).absolute()
    
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
        return {
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dim": self.EMBEDDING_DIM,
            "llm_model": self.LLM_MODEL,
            "vision_model": self.VISION_MODEL
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