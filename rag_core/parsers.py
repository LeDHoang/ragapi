from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import os
import subprocess, shutil
from rag_core.config import AppConfig
import polars as pl

class BaseParser:
    def parse_pdf(self, pdf_path: Path, **kw) -> List[Dict[str,Any]]:
        raise NotImplementedError
    def parse_image(self, image_path: Path, **kw) -> List[Dict[str,Any]]:
        raise NotImplementedError
    def parse_office_doc(self, doc_path: Path, **kw) -> List[Dict[str,Any]]:
        raise NotImplementedError
    def parse_document(self, file_path: Path, **kw) -> List[Dict[str,Any]]:
        raise NotImplementedError

# ------------ Docling wrapper (correct APIs for tables/pictures/pages)
class DoclingParser(BaseParser):
    def __init__(self, cfg: AppConfig):
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling_core.types.doc import PictureItem  # type marker for figures
            self._ok = True
            self._DocumentConverter = DocumentConverter
            self._InputFormat = InputFormat
            self._PdfFormatOption = PdfFormatOption
            self._PdfPipelineOptions = PdfPipelineOptions
            self._PictureItem = PictureItem
        except Exception:
            self._ok = False
        # store config
        self._cfg = cfg

    def _ensure(self):
        if not self._ok:
            raise RuntimeError("Docling not installed. `pip install docling`")

    def _mk_converter(self, want_page_images: bool = False, want_picture_images: bool = False):
        # Enable image generation so we can export page renders/pictures
        pipeline_options = self._PdfPipelineOptions()
        # Page renders are expensive; keep them off by default
        pipeline_options.generate_page_images = bool(want_page_images)
        # Figure images are controlled by config flag
        pipeline_options.generate_picture_images = bool(want_picture_images)
        # pipeline_options.images_scale = 2.0  # uncomment for higher DPI
        fmt = {
            self._InputFormat.PDF: self._PdfFormatOption(pipeline_options=pipeline_options)
        }
        return self._DocumentConverter(format_options=fmt)

    def _to_blocks(self, conv_res) -> List[Dict[str, Any]]:
        doc = conv_res.document
        blocks: List[Dict[str, Any]] = []

        # Get page sizes for bbox calculations
        page_sizes = {}
        pages = getattr(doc, "pages", None)
        if pages:
            for pg_no, page in pages.items():
                try:
                    # Get page dimensions from the page object
                    page_sizes[pg_no] = (page.size.width, page.size.height)
                except:
                    page_sizes[pg_no] = (612, 792)  # Default A4 size in points

        # 1) TEXT with enhanced metadata
        for i, t in enumerate(getattr(doc, "texts", [])):
            if t.text and t.text.strip():
                # Extract page info - fallback to sequential assignment if not available
                page_idx = getattr(t, "page_no", None)
                if page_idx is None:
                    # Fallback: estimate page based on position in document
                    # This is a rough approximation when page info is not available
                    page_idx = i // 50  # Assume ~50 text items per page

                # Extract bounding box if available
                bbox = None
                page_size = page_sizes.get(page_idx, None)
                try:
                    if hasattr(t, 'prov') and t.prov:
                        # Try to get bounding box from provenance
                        if hasattr(t.prov, 'bbox'):
                            bbox = (t.prov.bbox.x1, t.prov.bbox.y1, t.prov.bbox.x2, t.prov.bbox.y2)
                except:
                    pass

                # Extract headers if available
                headers = []
                if hasattr(t, 'level') and t.level > 0:
                    # Convert level to header markers
                    headers = ['#' * t.level]

                block = {
                    "type": "text",
                    "text": t.text,
                    "page_idx": page_idx,
                    "bbox": bbox,
                    "page_size": page_size
                }

                if headers:
                    block["headers"] = headers

                blocks.append(block)

        # 2) TABLES with enhanced metadata
        for i, tb in enumerate(getattr(doc, "tables", [])):
            try:
                body_md = tb.export_to_markdown()
            except Exception:
                body_md = (tb.export_to_html(doc=doc) or "")[:100000]

            # Extract page info - fallback to sequential assignment if not available
            page_idx = getattr(tb, "page_no", None)
            if page_idx is None:
                # Fallback: estimate page based on position in document
                page_idx = i // 5  # Assume fewer tables than text items

            page_size = page_sizes.get(page_idx, None)

            # Extract bounding box if available
            bbox = None
            try:
                if hasattr(tb, 'prov') and tb.prov:
                    if hasattr(tb.prov, 'bbox'):
                        bbox = (tb.prov.bbox.x1, tb.prov.bbox.y1, tb.prov.bbox.x2, tb.prov.bbox.y2)
            except:
                pass

            caps = getattr(tb, "captions", None)
            footnotes = getattr(tb, "footnotes", None)

            block = {
                "type": "table",
                "table_body": body_md,
                "table_caption": caps if isinstance(caps, list) else ([caps] if caps else []),
                "table_footnote": footnotes if isinstance(footnotes, list) else ([footnotes] if footnotes else []),
                "page_idx": page_idx,
                "bbox": bbox,
                "page_size": page_size
            }

            blocks.append(block)

        # 3) PICTURES with enhanced metadata
        from docling_core.types.doc import PictureItem
        for i, (element, _level) in enumerate(doc.iterate_items()):
            if isinstance(element, PictureItem):
                # Extract page info - fallback to sequential assignment if not available
                page_idx = getattr(element, "page_no", None)
                if page_idx is None:
                    # Fallback: estimate page based on position in document
                    page_idx = i // 10  # Assume fewer images than text items

                page_size = page_sizes.get(page_idx, None)

                # Extract bounding box if available
                bbox = None
                try:
                    if hasattr(element, 'prov') and element.prov:
                        if hasattr(element.prov, 'bbox'):
                            bbox = (element.prov.bbox.x1, element.prov.bbox.y1, element.prov.bbox.x2, element.prov.bbox.y2)
                except:
                    pass

                img_path = ""
                try:
                    pil_img = element.get_image(doc)
                    out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"docling-figure-p{page_idx or 0}-{id(element)}.png"
                    fpath = out_dir / fname
                    pil_img.save(fpath, "PNG")
                    img_path = str(fpath)
                except Exception:
                    pass

                caps = getattr(element, "captions", None)
                footnotes = getattr(element, "footnotes", None)

                block = {
                    "type": "image",
                    "img_path": img_path,
                    "image_caption": caps if isinstance(caps, list) else ([caps] if caps else []),
                    "image_footnote": footnotes if isinstance(footnotes, list) else ([footnotes] if footnotes else []),
                    "page_idx": page_idx,
                    "bbox": bbox,
                    "page_size": page_size
                }

                blocks.append(block)

        # 4) EQUATIONS - try to extract from document
        try:
            for i, eq in enumerate(getattr(doc, "equations", [])):
                # Extract page info - fallback to sequential assignment if not available
                page_idx = getattr(eq, "page_no", None)
                if page_idx is None:
                    # Fallback: estimate page based on position in document
                    page_idx = i // 20  # Assume fewer equations than text items

                page_size = page_sizes.get(page_idx, None)

                # Extract bounding box if available
                bbox = None
                try:
                    if hasattr(eq, 'prov') and eq.prov:
                        if hasattr(eq.prov, 'bbox'):
                            bbox = (eq.prov.bbox.x1, eq.prov.bbox.y1, eq.prov.bbox.x2, eq.prov.bbox.y2)
                except:
                    pass

                # Get equation text and format
                eq_text = ""
                eq_format = "plain"

                if hasattr(eq, 'text') and eq.text:
                    eq_text = eq.text
                    if hasattr(eq, 'format'):
                        eq_format = eq.format
                    elif "\\" in eq_text or "$" in eq_text:
                        eq_format = "latex"

                if eq_text:
                    caps = getattr(eq, "captions", None)
                    block = {
                        "type": "equation",
                        "text": eq_text,
                        "text_format": eq_format,
                        "page_idx": page_idx,
                        "bbox": bbox,
                        "page_size": page_size
                    }

                    if caps:
                        block["equation_caption"] = caps if isinstance(caps, list) else [caps]

                    blocks.append(block)
        except:
            pass

        # 5) PAGE RENDERS (optional)
        pages = getattr(doc, "pages", None)
        if pages:
            for _pg_no, page in pages.items():
                try:
                    pil = page.image.pil_image
                    out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fpath = out_dir / f"docling-page-{page.page_no}.png"
                    pil.save(fpath, "PNG")
                    blocks.append({
                        "type": "image",
                        "img_path": str(fpath),
                        "image_caption": [f"Rendered page {page.page_no}"],
                        "page_idx": page.page_no,
                        "bbox": None,
                        "page_size": page_sizes.get(page.page_no, None)
                    })
                except Exception:
                    pass

        if not blocks:
            md = doc.export_to_markdown()
            blocks.append({
                "type": "text",
                "text": md[:200000],
                "page_idx": 0,
                "bbox": None,
                "page_size": None
            })

        # Sort blocks by page and position for consistent ordering
        blocks.sort(key=lambda x: (
            x.get("page_idx", 0),
            x.get("bbox", [0, 0, 0, 0])[1] if x.get("bbox") else 0,  # y1 coordinate
            x.get("bbox", [0, 0, 0, 0])[0] if x.get("bbox") else 0   # x1 coordinate
        ))

        return blocks

    def parse_pdf(self, pdf_path: Path, **kw) -> List[Dict[str, Any]]:
        self._ensure()
        # Do not generate page renders; optionally include figure images if enabled
        converter = self._mk_converter(
            want_page_images=False,
            want_picture_images=bool(self._cfg.enable_image_processing)
        )
        conv_res = converter.convert(str(pdf_path))
        return self._to_blocks(conv_res)

    def parse_office_doc(self, doc_path: Path, **kw) -> List[Dict[str, Any]]:
        self._ensure()
        converter = self._mk_converter(
            want_page_images=False,
            want_picture_images=bool(self._cfg.enable_image_processing)
        )
        conv_res = converter.convert(str(doc_path))
        return self._to_blocks(conv_res)

    def parse_image(self, image_path: Path, **kw) -> List[Dict[str, Any]]:
        return [{"type": "image", "img_path": str(image_path), "page_idx": 0}]

    def parse_document(self, file_path: Path, **kw) -> List[Dict[str, Any]]:
        return self.parse_office_doc(file_path, **kw)

# ------------ MinerU wrapper (sketch)
class MineruParser(BaseParser):
    def __init__(self):
        try:
            import mineru  # noqa
            self._ok = True
        except Exception:
            self._ok = False

    def _ensure(self):
        if not self._ok:
            raise RuntimeError("MinerU not installed. `pip install mineru`")

    def parse_pdf(self, pdf_path: Path, **kw) -> List[Dict[str,Any]]:
        self._ensure()
        raise NotImplementedError("Hook MinerU PDF here")

    def parse_image(self, image_path: Path, **kw) -> List[Dict[str,Any]]:
        self._ensure()
        return [{"type":"image","img_path": str(image_path)}]

    def parse_office_doc(self, doc_path: Path, **kw) -> List[Dict[str,Any]]:
        self._ensure()
        if not shutil.which("soffice"):
            raise RuntimeError("LibreOffice required for MinerU Office parsing")
        pdf_out = doc_path.with_suffix(".pdf")
        subprocess.run(
            ["soffice","--headless","--convert-to","pdf",str(doc_path),"--outdir",str(doc_path.parent)],
            check=True
        )
        return self.parse_pdf(pdf_out, **kw)

    def parse_document(self, file_path: Path, **kw) -> List[Dict[str,Any]]:
        return [{"type":"text","text": file_path.read_text(errors="ignore")[:10000]}]

# ------------ Excel-native “Path B” summarizer for huge sheets
def excel_native_summary(path: Path, max_preview_rows: int = 20) -> List[Dict[str,Any]]:
    items: List[Dict[str,Any]] = []
    try:
        x = pl.read_excel(str(path))  # first sheet
        dtypes = [f"{col}: {x[col].dtype}" for col in x.columns]
        nulls = [f"{col}: {int(x[col].null_count())}" for col in x.columns]
        dict_text = "Columns & types:\n- " + "\n- ".join(dtypes) + "\nNull counts:\n- " + "\n- ".join(nulls)
        items.append({"type":"text","text": dict_text})
        preview = x.head(max_preview_rows)
        body = preview.to_pandas().to_markdown(index=False)
        items.append({"type":"table","table_body": body, "table_caption":["Excel preview table"]})
    except Exception as e:
        items.append({"type":"text","text": f"Excel summary failed: {e}"})
    return items

def make_parser(cfg: AppConfig) -> BaseParser:
    if cfg.parser.lower() == "docling":
        return DoclingParser(cfg)
    return MineruParser()
