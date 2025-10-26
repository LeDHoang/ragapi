from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
from rag_core.config import AppConfig
from rag_core.llm_unified import UnifiedLLM
from rag_core.context_extractor import ContextExtractor

SYSTEM_CAPTION = (
    "You help create short, faithful captions/summaries for RAG chunks. "
    "Keep them concise and factual. Do not invent data."
)

SYSTEM_IMAGE_ANALYSIS = (
    "You are an expert image analyst. Analyze this image considering the surrounding context. "
    "Provide a comprehensive but concise description that captures the key visual elements, "
    "data, and insights. If text is visible in the image, include important textual content."
)

SYSTEM_TABLE_ANALYSIS = (
    "You are an expert data analyst. Analyze this table considering the surrounding context. "
    "Summarize the key findings, trends, and important data points. Include relevant metrics, "
    "units, and time ranges when present."
)

SYSTEM_EQUATION_ANALYSIS = (
    "You are a mathematics expert. Analyze this equation considering the surrounding context. "
    "Explain what this equation represents and its key components. If this is part of a larger "
    "mathematical concept or derivation, explain the context."
)

@dataclass
class ModalProcessors:
    cfg: AppConfig
    llm: Optional[UnifiedLLM] = None
    context_extractor: Optional[ContextExtractor] = None

    @staticmethod
    def from_config(cfg: AppConfig, llm: Optional[UnifiedLLM] = None, tokenizer=None) -> "ModalProcessors":
        context_extractor = ContextExtractor(cfg, tokenizer) if cfg else None
        return ModalProcessors(cfg, llm, context_extractor)

    async def describe_item(self, item: Dict[str,Any], context: Optional[str] = None) -> str:
        t = item.get("type","text")

        # Use LLM for multimodal content if available
        if self.llm:
            if t == "image":
                return await self._describe_image_with_llm(item, context)
            elif t == "table":
                return await self._describe_table_with_llm(item, context)
            elif t == "equation":
                return await self._describe_equation_with_llm(item, context)

        # Fallback descriptions
        return self._get_fallback_description(item)

    async def _describe_image_with_llm(self, item: Dict[str,Any], context: Optional[str] = None) -> str:
        """Generate image description using LLM with vision capabilities"""
        if not self.llm:
            return self._get_fallback_description(item)

        # Check if image path exists
        img_path = item.get("img_path")
        if not img_path or not Path(img_path).exists():
            return f"Image on page {item.get('page_idx', 0)} (file not found)."

        # Build prompt with context
        prompt_parts = ["Analyze this image:"]

        # Add captions and footnotes
        captions = item.get('image_caption', [])
        footnotes = item.get('image_footnote', [])
        if captions:
            prompt_parts.append(f"Caption: {'; '.join(captions)}")
        if footnotes:
            prompt_parts.append(f"Footnote: {'; '.join(footnotes)}")

        # Add surrounding context
        if context:
            prompt_parts.append(f"Context: {context}")

        prompt_parts.append("Provide a detailed but concise visual description focusing on:")
        prompt_parts.extend([
            "- Main subject and key elements",
            "- Any visible text or data",
            "- Visual characteristics and layout",
            "- Important details for understanding the content"
        ])

        user_prompt = "\n".join(prompt_parts)

        try:
            # Use vision-capable LLM
            return self.llm.generate_with_image(SYSTEM_IMAGE_ANALYSIS, user_prompt, img_path)
        except Exception as e:
            # Fallback if vision fails
            return self._get_fallback_description(item)

    async def _describe_table_with_llm(self, item: Dict[str,Any], context: Optional[str] = None) -> str:
        """Generate table description using LLM"""
        if not self.llm:
            return self._get_fallback_description(item)

        body = item.get("table_body","")

        # Truncate if too long
        if len(body) > self.cfg.table_body_max_chars:
            body = body[:self.cfg.table_body_max_chars] + "..."

        prompt_parts = ["Analyze this table data:"]

        # Add caption
        captions = item.get('table_caption', [])
        if captions:
            prompt_parts.append(f"Caption: {'; '.join(captions)}")

        # Add context
        if context:
            prompt_parts.append(f"Context: {context}")

        prompt_parts.append("Table data:")
        prompt_parts.append(body)
        prompt_parts.append("\nSummarize the key findings and insights from this table.")

        user_prompt = "\n".join(prompt_parts)

        try:
            return self.llm.generate(SYSTEM_TABLE_ANALYSIS, user_prompt)
        except Exception as e:
            return self._get_fallback_description(item)

    async def _describe_equation_with_llm(self, item: Dict[str,Any], context: Optional[str] = None) -> str:
        """Generate equation description using LLM"""
        if not self.llm:
            return self._get_fallback_description(item)

        eq_text = item.get("text", "")
        eq_format = item.get("text_format", "plain")

        prompt_parts = ["Analyze this mathematical expression:"]

        # Add context
        if context:
            prompt_parts.append(f"Context: {context}")

        prompt_parts.append(f"Expression ({eq_format}): {eq_text}")
        prompt_parts.append("\nExplain what this equation represents and its significance.")

        user_prompt = "\n".join(prompt_parts)

        try:
            return self.llm.generate(SYSTEM_EQUATION_ANALYSIS, user_prompt)
        except Exception as e:
            return self._get_fallback_description(item)

    def _get_fallback_description(self, item: Dict[str,Any]) -> str:
        """Generate fallback descriptions when LLM is not available"""
        t = item.get("type","text")

        if t == "image":
            page = item.get('page_idx', 0)
            caps = item.get('image_caption', [])
            cap_text = '; '.join(caps) if caps else 'No caption'
            return f"Figure on page {page + 1}: {cap_text}"

        if t == "table":
            body = item.get("table_body","")
            rows = len(body.splitlines()) if body else 0
            caps = item.get('table_caption', [])
            cap_text = '; '.join(caps) if caps else 'No caption'
            return f"Table on page {item.get('page_idx', 0) + 1}: {cap_text} ({rows} rows)"

        if t == "equation":
            eq_text = item.get("text", "")
            return f"Equation on page {item.get('page_idx', 0) + 1}: {eq_text[:100]}"

        if t == "text":
            txt = item.get("text","")
            return f"Text on page {item.get('page_idx', 0) + 1} ({len(txt)} chars)"

        return f"{t.title()} content"
