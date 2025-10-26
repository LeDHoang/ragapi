from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from rag_core.config import AppConfig
from rag_core.llm_bedrock import BedrockLLM

SYSTEM_CAPTION = (
    "You help create short, faithful captions/summaries for RAG chunks. "
    "Keep them concise and factual. Do not invent data."
)

@dataclass
class ModalProcessors:
    cfg: AppConfig
    llm: Optional[BedrockLLM] = None

    @staticmethod
    def from_config(cfg: AppConfig, llm: Optional[BedrockLLM] = None) -> "ModalProcessors":
        return ModalProcessors(cfg, llm)

    async def describe_item(self, item: Dict[str,Any]) -> str:
        t = item.get("type","text")
        
        # Use LLM only for tables and images
        if t == "image" and self.llm:
            prompt = "Provide a one-sentence factual caption for this figure. If caption/footnote exist, refine them."
            meta = f"captions={item.get('image_caption', [])} footnotes={item.get('image_footnote', [])}"
            return self.llm.generate(SYSTEM_CAPTION, f"{prompt}\n\n{meta}")
        elif t == "table" and self.llm:
            body = item.get("table_body","")
            prompt = (
                "Summarize the key facts from this table in one sentence. "
                "Mention metrics/units/time range if present."
            )
            return self.llm.generate(SYSTEM_CAPTION, f"{prompt}\n\n{body[:4000]}")
        
        # Use simple fallbacks for all other types
        if t == "image":
            return f"Figure on page {item.get('page_idx', 0)}."
        if t == "table":
            body = item.get("table_body","")
            rows = len(body.splitlines())
            return f"Table with ~{rows} lines (markdown preview)."
        if t == "equation":
            return "Mathematical expression."
        if t == "text":
            txt = item.get("text","")
            return f"Text snippet ({len(txt)} chars)."
        return "Content snippet."
