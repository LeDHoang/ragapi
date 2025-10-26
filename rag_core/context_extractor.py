from __future__ import annotations
from typing import List, Dict, Any, Optional
from rag_core.config import AppConfig
import re

class ContextExtractor:
    """Extract context around multimodal content for better descriptions"""

    def __init__(self, cfg: AppConfig, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def extract_context(self, content_source: List[Dict[str, Any]], current_item: Dict[str, Any],
                       content_format: str = "auto") -> str:
        """Extract context around current multimodal item"""

        current_page = current_item.get("page_idx")
        if current_page is None:
            # If no page info, return empty context
            return ""

        window_size = self.cfg.context_window
        start_page = max(0, current_page - window_size)
        end_page = current_page + window_size + 1

        context_parts = []

        for item in content_source:
            item_page = item.get("page_idx")
            if item_page is None:
                continue
            item_type = item.get("type", "")

            # Check if within context window and matches filter
            if (start_page <= item_page < end_page and
                item_type in self.cfg.context_filter_content_types):

                text_content = self._extract_text_from_item(item)
                if text_content:
                    if item_page != current_page:
                        context_parts.append(f"[Page {item_page + 1}] {text_content}")
                    else:
                        context_parts.append(text_content)

        context_text = "\n".join(context_parts)

        # Truncate by tokens if tokenizer available
        if self.tokenizer and self.cfg.max_context_tokens > 0:
            try:
                tokens = self.tokenizer.encode(context_text)
                if len(tokens) > self.cfg.max_context_tokens:
                    context_text = self.tokenizer.decode(tokens[:self.cfg.max_context_tokens])
            except:
                # Fallback to character limit
                if len(context_text) > self.cfg.max_context_tokens * 4:  # Rough token estimation
                    context_text = context_text[:self.cfg.max_context_tokens * 4]

        return context_text

    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """Extract text content from different item types"""
        item_type = item.get("type", "")

        if item_type == "text":
            text = item.get("text", "")
            # Include headers if available and enabled
            if self.cfg.include_headers and item.get("headers"):
                headers = " > ".join(item["headers"])
                text = f"{headers}\n{text}"
            return text

        elif item_type == "table":
            parts = []
            if self.cfg.include_captions and item.get("table_caption"):
                parts.extend(item["table_caption"])
            if item.get("table_body"):
                # Extract first few rows for context
                lines = item["table_body"].split('\n')[:5]
                parts.append("Table data: " + " | ".join(lines))
            return " | ".join(parts)

        elif item_type == "image":
            parts = []
            if self.cfg.include_captions and item.get("image_caption"):
                parts.extend(item["image_caption"])
            if item.get("image_footnote"):
                parts.extend(item["image_footnote"])
            return " | ".join(parts)

        elif item_type == "equation":
            text = item.get("text", "")
            if item.get("text_format") == "latex":
                return f"LaTeX equation: {text}"
            return f"Equation: {text}"

        return ""

    def extract_latex_equations(self, text_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract LaTeX equations from text items and create separate equation items"""
        equations = []

        for item in text_items:
            text = item.get("text", "")
            if not text:
                continue

            # Find LaTeX patterns
            patterns = [
                (r'\$\$(.*?)\$\$', 'display'),  # Display math
                (r'\\\[(.*?)\\\]', 'display'),  # Display math alternative
                (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),  # Equation environment
                (r'\$(.*?)\$', 'inline')        # Inline math
            ]

            for pattern, math_type in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    # Create equation item
                    equation_item = {
                        "type": "equation",
                        "text": match.strip(),
                        "text_format": "latex",
                        "page_idx": item.get("page_idx"),
                        "bbox": item.get("bbox"),
                        "page_size": item.get("page_size")
                    }

                    # Estimate position in text (rough approximation)
                    match_start = text.find(match)
                    if match_start >= 0 and item.get("bbox"):
                        # This is a very rough approximation - in practice you'd want
                        # the actual text position within the page
                        pass

                    equations.append(equation_item)

        return equations

    def build_page_context_map(self, content_list: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Build a map of page index to content items for efficient context extraction"""
        page_map = {}
        for item in content_list:
            page_idx = item.get("page_idx")
            if page_idx is None:
                # Skip items without page information
                continue

            if page_idx not in page_map:
                page_map[page_idx] = []
            page_map[page_idx].append(item)

        # Sort items within each page by their position (if bbox available)
        for page_idx in page_map:
            items = page_map[page_idx]
            # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
            items.sort(key=lambda x: (
                (x.get("bbox", [0, 0, 0, 0])[1] if x.get("bbox") else 0),  # y1 coordinate
                (x.get("bbox", [0, 0, 0, 0])[0] if x.get("bbox") else 0)   # x1 coordinate
            ))

        return page_map

def create_context_extractor(cfg: AppConfig, tokenizer=None) -> ContextExtractor:
    """Factory function to create context extractor"""
    return ContextExtractor(cfg, tokenizer)
