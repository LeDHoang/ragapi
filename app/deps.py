# app/deps.py
from functools import lru_cache
from rag_core.config import AppConfig
from rag_core.storage import StorageBundle
from rag_core.parsers import make_parser
from rag_core.pipeline import RagPipeline
from rag_core.processors import ModalProcessors
from rag_core.llm_bedrock import BedrockLLM

@lru_cache()
def get_config() -> AppConfig:
    return AppConfig.from_env()

@lru_cache()
def get_storages() -> StorageBundle:
    cfg = get_config()
    return StorageBundle.initialize(cfg)

@lru_cache()
def get_llm() -> BedrockLLM:
    # Reads BEDROCK_REGION, BEDROCK_LLM_MODEL_ID, and AWS_BEARER_TOKEN_BEDROCK from .env
    return BedrockLLM()

@lru_cache()
def get_processors() -> ModalProcessors:
    cfg = get_config()
    llm = get_llm()
    return ModalProcessors.from_config(cfg, llm=llm)

def get_pipeline() -> RagPipeline:
    cfg = get_config()
    stores = get_storages()
    procs = get_processors()
    parser = make_parser(cfg)
    llm = get_llm()
    return RagPipeline(cfg, stores, parser, procs, llm=llm)
