from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from rag_core.config import AppConfig

class LayoutOverlay:
    """Generate layout overlay images with bounding boxes for parsed content"""

    # Color scheme for different content types
    COLORS = {
        "text": (0, 0, 255, 128),      # Blue with transparency
        "image": (0, 255, 0, 128),     # Green with transparency
        "table": (255, 165, 0, 128),   # Orange with transparency
        "equation": (255, 0, 255, 128) # Purple with transparency
    }

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.overlay_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, pdf_path: Path, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render layout overlay for all pages in the PDF"""
        doc = fitz.open(str(pdf_path))
        results = {
            "pages": [],
            "manifest": []
        }

        for page_num in range(len(doc)):
            page_content = [item for item in content_list if item.get("page_idx") == page_num]

            if not page_content:
                continue

            # Render page to image
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(self.cfg.overlay_dpi / 72, self.cfg.overlay_dpi / 72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Draw bounding boxes
            self._draw_bboxes(img, page_content, page_num)

            # Save overlay image
            overlay_path = self.output_dir / f"{pdf_path.stem}_page_{page_num + 1}_overlay.png"
            img.save(overlay_path)

            # Create manifest entry
            page_manifest = {
                "page_idx": page_num,
                "overlay_path": str(overlay_path),
                "page_size": (pix.width, pix.height),
                "elements": []
            }

            for item in page_content:
                if item.get("bbox"):
                    page_manifest["elements"].append({
                        "type": item["type"],
                        "bbox": item["bbox"],
                        "chunk_id": item.get("chunk_id")
                    })

            results["pages"].append(page_manifest)
            results["manifest"].extend(page_manifest["elements"])

        doc.close()

        # Save manifest file
        manifest_path = self.output_dir / f"{pdf_path.stem}_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return {
            "manifest_path": str(manifest_path),
            "overlay_dir": str(self.output_dir),
            "total_pages": len(results["pages"]),
            "total_elements": len(results["manifest"])
        }

    def _draw_bboxes(self, img: Image.Image, page_content: List[Dict[str, Any]], page_idx: int):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(img)

        # Try to use a font, fallback to default
        try:
            font = ImageFont.load_default()
        except:
            font = None

        for item in page_content:
            bbox = item.get("bbox")
            if not bbox:
                continue

            item_type = item.get("type", "text")
            color = self.COLORS.get(item_type, (128, 128, 128, 128))

            # Scale bbox from PDF coordinates to image coordinates
            x1, y1, x2, y2 = bbox
            page_size = item.get("page_size")
            if page_size:
                # PDF coordinates: (0,0) at bottom-left, image coordinates: (0,0) at top-left
                img_width, img_height = img.size
                pdf_width, pdf_height = page_size

                # Scale factors
                scale_x = img_width / pdf_width
                scale_y = img_height / pdf_height

                # Convert coordinates
                img_x1 = x1 * scale_x
                img_y1 = img_height - (y2 * scale_y)  # Flip Y coordinate
                img_x2 = x2 * scale_x
                img_y2 = img_height - (y1 * scale_y)  # Flip Y coordinate

                # Draw rectangle
                draw.rectangle([img_x1, img_y1, img_x2, img_y2], outline=color[:3], fill=color, width=2)

                # Add label
                label = item_type.upper()
                if font:
                    # Get text size for positioning
                    text_bbox = draw.textbbox((img_x1, img_y1 - 20), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # Draw background rectangle for text
                    draw.rectangle([img_x1, img_y1 - 20, img_x1 + text_width, img_y1], fill=color[:3])

                    # Draw text
                    draw.text((img_x1, img_y1 - 20), label, fill="white", font=font)

    def render_legend(self) -> Image.Image:
        """Generate a legend image explaining the color scheme"""
        img_width, img_height = 300, 200
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Title
        if font:
            draw.text((10, 10), "Layout Overlay Legend", fill="black", font=font)

        y_offset = 40
        for item_type, color in self.COLORS.items():
            # Draw color swatch
            draw.rectangle([10, y_offset, 30, y_offset + 20], fill=color[:3])

            # Draw label
            if font:
                draw.text((40, y_offset), item_type.title(), fill="black", font=font)

            y_offset += 30

        # Save legend
        legend_path = self.output_dir / "legend.png"
        img.save(legend_path)

        return img

def create_overlay_generator(cfg: AppConfig) -> LayoutOverlay:
    """Factory function to create overlay generator"""
    return LayoutOverlay(cfg)
