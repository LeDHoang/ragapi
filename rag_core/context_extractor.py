from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from .config import config
from .schemas import ContentType

logger = logging.getLogger(__name__)

class ContextExtractor:
    def __init__(self, config_params: Optional[Dict[str, Any]] = None):
        self.config = config_params or config.get_processing_config()
        self.context_window = 2  # Pages before/after for context
        self.max_context_tokens = 1000  # Max tokens for context
    
    def extract_context(
        self,
        content_source: List[Dict[str, Any]],
        current_item: Dict[str, Any],
        content_format: str = "auto"
    ) -> str:
        """Extract context around current multimodal item"""
        
        current_page = current_item.get("page_idx", 0)
        window_size = self.context_window
        
        start_page = max(0, current_page - window_size)
        end_page = current_page + window_size + 1
        
        context_parts = []
        
        for item in content_source:
            item_page = item.get("page_idx", 0)
            item_type = item.get("type", "")
            
            # Check if within context window
            if start_page <= item_page < end_page:
                text_content = self._extract_text_from_item(item)
                if text_content:
                    if item_page != current_page:
                        context_parts.append(f"[Page {item_page}] {text_content}")
                    else:
                        context_parts.append(text_content)
        
        return "\n".join(context_parts)
    
    def _extract_text_from_item(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract text content from different item types"""
        
        item_type = item.get("type", "")
        
        if item_type == ContentType.TEXT:
            return item.get("text", "")
        elif item_type == ContentType.IMAGE:
            captions = item.get("image_caption", [])
            footnotes = item.get("image_footnote", [])
            parts = []
            if captions:
                parts.append(f"Image Caption: {' '.join(captions)}")
            if footnotes:
                parts.append(f"Image Footnote: {' '.join(footnotes)}")
            return " | ".join(parts) if parts else None
        elif item_type == ContentType.TABLE:
            captions = item.get("table_caption", [])
            return f"Table: {' '.join(captions)}" if captions else None
        elif item_type == ContentType.EQUATION:
            return item.get("text", None)
        return None

    def analyze_document_structure(
        self,
        content_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze document structure and extract metadata"""
        
        structure = {
            "total_pages": 0,
            "content_types": {},
            "heading_structure": [],
            "sections": []
        }
        
        current_section = None
        
        for item in content_list:
            # Update page count
            page_idx = item.get("page_idx", 0)
            structure["total_pages"] = max(structure["total_pages"], page_idx + 1)
            
            # Count content types
            content_type = item.get("type", "unknown")
            structure["content_types"][content_type] = structure["content_types"].get(content_type, 0) + 1
            
            # Analyze text structure
            if content_type == ContentType.TEXT:
                text_level = item.get("text_level", 0)
                text = item.get("text", "").strip()
                
                if text_level > 0:  # This is a heading
                    structure["heading_structure"].append({
                        "level": text_level,
                        "text": text,
                        "page": page_idx
                    })
                    
                    # Start new section
                    if current_section:
                        structure["sections"].append(current_section)
                    
                    current_section = {
                        "heading": text,
                        "level": text_level,
                        "start_page": page_idx,
                        "content": []
                    }
                
                # Add to current section if exists
                if current_section:
                    current_section["content"].append(item)
        
        # Add last section
        if current_section:
            structure["sections"].append(current_section)
        
        return structure

    def get_section_context(
        self,
        content_list: List[Dict[str, Any]],
        target_page: int
    ) -> Optional[Dict[str, Any]]:
        """Get section context for a specific page"""
        
        structure = self.analyze_document_structure(content_list)
        
        for section in structure["sections"]:
            # Find content from this section on target page
            section_content = [
                item for item in section["content"]
                if item.get("page_idx", 0) == target_page
            ]
            
            if section_content:
                return {
                    "section_heading": section["heading"],
                    "section_level": section["level"],
                    "section_content": section_content
                }
        
        return None

    def extract_references(
        self,
        content_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract cross-references between content items"""
        
        references = []
        
        for i, item1 in enumerate(content_list):
            item1_page = item1.get("page_idx", 0)
            item1_type = item1.get("type", "")
            
            for j, item2 in enumerate(content_list[i+1:], i+1):
                item2_page = item2.get("page_idx", 0)
                item2_type = item2.get("type", "")
                
                # Check for potential references
                if abs(item1_page - item2_page) <= self.context_window:
                    if self._items_are_related(item1, item2):
                        references.append({
                            "source": self._get_item_identifier(item1),
                            "target": self._get_item_identifier(item2),
                            "type": "contextual",
                            "confidence": 0.8
                        })
        
        return references
    
    def _items_are_related(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Check if two items are potentially related"""
        # This is a simplified check - in practice you'd want more sophisticated logic
        
        # Same page items are likely related
        if item1.get("page_idx") == item2.get("page_idx"):
            return True
        
        # Check for caption references
        if item1.get("type") == ContentType.IMAGE:
            captions = item1.get("image_caption", [])
            if any(self._text_references_item(caption, item2) for caption in captions):
                return True
        
        if item2.get("type") == ContentType.IMAGE:
            captions = item2.get("image_caption", [])
            if any(self._text_references_item(caption, item1) for caption in captions):
                return True
        
        return False
    
    def _text_references_item(self, text: str, item: Dict[str, Any]) -> bool:
        """Check if text references another item"""
        # This is a simplified check - in practice you'd want more sophisticated logic
        if item.get("type") == ContentType.IMAGE:
            return "figure" in text.lower() or "image" in text.lower()
        elif item.get("type") == ContentType.TABLE:
            return "table" in text.lower()
        return False
    
    def _get_item_identifier(self, item: Dict[str, Any]) -> str:
        """Get a unique identifier for an item"""
        item_type = item.get("type", "unknown")
        page = item.get("page_idx", 0)
        
        if item_type == ContentType.IMAGE:
            return f"image_{page}_{Path(item.get('img_path', '')).stem}"
        elif item_type == ContentType.TABLE:
            caption = "_".join(item.get("table_caption", [])[:50])
            return f"table_{page}_{caption}"
        elif item_type == ContentType.TEXT:
            text = item.get("text", "")[:50]
            return f"text_{page}_{text}"
        
        return f"{item_type}_{page}"