from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json
import base64
from PIL import Image
import io

from .config import config
from .schemas import ContentType
from .context_extractor import ContextExtractor

logger = logging.getLogger(__name__)

class ContentProcessor:
    def __init__(self):
        self.config = config
        self.context_extractor = ContextExtractor()
    
    def separate_content(
        self,
        content_list: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Separate text and multimodal content"""
        
        text_parts = []
        multimodal_items = []
        
        for item in content_list:
            raw_type = item.get("type", "text")
            content_type = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
            
            if content_type == ContentType.TEXT.value:
                text_content = item.get("text", "").strip()
                if text_content:
                    # Add heading markers for structure
                    text_level = item.get("text_level", 0)
                    if text_level > 0:
                        text_content = f"{'#' * text_level} {text_content}"
                    
                    text_parts.append(text_content)
            else:
                # Process and enhance multimodal content
                enhanced_item = self._enhance_multimodal_item(item)
                if enhanced_item:
                    multimodal_items.append(enhanced_item)
        
        # Combine all text with double newlines
        full_text = "\n\n".join(text_parts)
        
        # Log content distribution
        type_counts = {}
        for item in multimodal_items:
            item_type = item.get("type", "unknown")
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        logger.info(f"Content separated: {len(full_text)} chars text, {len(multimodal_items)} multimodal items")
        logger.info(f"Multimodal distribution: {type_counts}")
        
        return full_text, multimodal_items
    
    def _enhance_multimodal_item(
        self,
        item: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Enhance multimodal content with additional context and metadata"""
        
        raw_type = item.get("type")
        content_type = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
        
        if content_type == ContentType.IMAGE.value:
            return self._enhance_image_item(item)
        elif content_type == ContentType.TABLE.value:
            return self._enhance_table_item(item)
        elif content_type == ContentType.EQUATION.value:
            return self._enhance_equation_item(item)
        
        return item
    
    def _enhance_image_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhance image content with additional metadata"""
        
        img_path = item.get("img_path")
        if not img_path or not Path(img_path).exists():
            return None
        
        try:
            # Get image metadata
            with Image.open(img_path) as img:
                width, height = img.size
                format = img.format
                mode = img.mode
            
            # Add metadata to item
            enhanced_item = item.copy()
            enhanced_item.update({
                "image_metadata": {
                    "width": width,
                    "height": height,
                    "format": format,
                    "mode": mode,
                    "aspect_ratio": width / height
                }
            })
            
            # Add base64 preview for small images
            if width * height < 1000000:  # Less than 1MP
                enhanced_item["image_preview"] = self._get_image_preview(img_path)
            
            return enhanced_item
            
        except Exception as e:
            logger.warning(f"Failed to enhance image {img_path}: {str(e)}")
            return item
    
    def _enhance_table_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance table content with additional metadata"""
        
        enhanced_item = item.copy()
        table_body = item.get("table_body", "")
        
        # Extract table dimensions
        rows = table_body.strip().split("\n")
        if rows:
            cols = len(rows[0].split("|"))
            enhanced_item["table_metadata"] = {
                "rows": len(rows),
                "columns": cols - 2  # Account for markdown table syntax
            }
        
        return enhanced_item
    
    def _enhance_equation_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance equation content with additional metadata"""
        
        enhanced_item = item.copy()
        latex = item.get("latex", "")
        
        # Add basic equation metadata
        enhanced_item["equation_metadata"] = {
            "length": len(latex),
            "complexity": self._estimate_equation_complexity(latex)
        }
        
        return enhanced_item
    
    def _get_image_preview(self, image_path: str, max_size: int = 300) -> str:
        """Create base64 preview of image"""
        
        try:
            with Image.open(image_path) as img:
                # Resize if needed
                if img.width > max_size or img.height > max_size:
                    ratio = min(max_size / img.width, max_size / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return base64.b64encode(buffer.getvalue()).decode()
                
        except Exception as e:
            logger.warning(f"Failed to create image preview: {str(e)}")
            return ""
    
    def _estimate_equation_complexity(self, latex: str) -> int:
        """Estimate equation complexity based on LaTeX content"""
        
        # This is a simple heuristic - you might want to make it more sophisticated
        complexity = 1
        
        # Special characters indicate complexity
        special_chars = ['\\sum', '\\int', '\\prod', '\\lim', '\\frac']
        for char in special_chars:
            if char in latex:
                complexity += 1
        
        # Nested brackets indicate complexity
        depth = 0
        max_depth = 0
        for char in latex:
            if char in '{([':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char in '})]':
                depth = max(0, depth - 1)
        
        complexity += max_depth
        
        return complexity

class ContentSeparator:
    def __init__(self):
        self.processor = ContentProcessor()
    
    async def process_document_content(
        self,
        content_list: List[Dict[str, Any]],
        doc_id: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Process and separate document content"""
        
        # Extract document structure
        context_extractor = ContextExtractor()
        doc_structure = context_extractor.analyze_document_structure(content_list)
        
        # Separate content
        full_text, multimodal_items = self.processor.separate_content(content_list)
        
        # Extract references between items
        references = context_extractor.extract_references(content_list)
        
        # Create processing summary
        summary = {
            "doc_id": doc_id,
            "structure": doc_structure,
            "text_length": len(full_text),
            "multimodal_count": len(multimodal_items),
            "references": references
        }
        
        return full_text, multimodal_items, summary