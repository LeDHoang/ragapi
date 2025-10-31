from typing import List, Dict, Any, Optional
from pathlib import Path
import subprocess
import json
import tempfile
import logging
import shutil
import sys
import re
from .config import config
from .schemas import ContentType, TextContent, ImageContent, TableContent, EquationContent

logger = logging.getLogger(__name__)

class BaseParser:
    def __init__(self):
        self.config = config.get_parser_config()
    
    async def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Base method for document parsing"""
        raise NotImplementedError
    
    @staticmethod
    def _persist_assets(
        content_list: List[Dict[str, Any]],
        source_root: Path,
        dest_root: Path
    ) -> List[Dict[str, Any]]:
        """Copy parser-produced assets (e.g., images) from a temp directory to a stable location
        and rewrite paths inside content_list to point to the persisted files.
        """
        dest_root.mkdir(parents=True, exist_ok=True)
        updated: List[Dict[str, Any]] = []
        
        for item in content_list:
            item_copy = dict(item)
            if item_copy.get("type") == ContentType.IMAGE:
                img_path = item_copy.get("img_path")
                if img_path:
                    p = Path(img_path)
                    if not p.is_absolute():
                        p = source_root / p
                    try:
                        if p.exists():
                            target = dest_root / p.name
                            if str(p.resolve()) != str(target.resolve()):
                                shutil.copy2(p, target)
                            item_copy["img_path"] = str(target)
                        else:
                            # leave as-is; downstream may skip if not exists
                            pass
                    except Exception as e:
                        logger.warning(f"Failed to persist image asset {p}: {e}")
            updated.append(item_copy)
        return updated

class MineruParser(BaseParser):
    async def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse document using MinerU"""

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert Office documents to PDF first, skip if already PDF
            from .conversion.excel_to_pdf import convert_office_to_pdf, needs_conversion
            if needs_conversion(file_path):
                logger.info("[MINERU] Converting Office document to PDF: %s", file_path)
                file_path, _ = convert_office_to_pdf(file_path, temp_dir)
            else:
                logger.debug("[MINERU] Skipping conversion for PDF file: %s", file_path)

            # Build MinerU command using the virtual environment executable
            mineru_cmd = str(Path(sys.executable).parent / "mineru")

            base_args = [
                "-p", file_path,
                "-o", temp_dir,
                "-m", self.config["method"],
                "--device", "cpu",
                "--lang", "en"
            ]
            if self.config["enable_equations"]:
                base_args.extend(["--formula", "true"])
            if self.config["enable_tables"]:
                base_args.extend(["--table", "true"])

            # Use the virtual environment mineru executable
            cmd = [mineru_cmd, *base_args]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"MinerU command failed: {result.stderr}")
            except FileNotFoundError as e:
                raise Exception(f"MinerU executable not found at {mineru_cmd}: {str(e)}")
            
            # Read output files
            stem = Path(file_path).stem

            # Persist full MinerU output for debugging/auditing
            mineru_temp_root = Path(temp_dir) / stem
            persist_output_root = config.get_working_dir() / "output" / stem
            if mineru_temp_root.exists():
                persist_output_root.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(mineru_temp_root, persist_output_root, dirs_exist_ok=True)
            else:
                persist_output_root.mkdir(parents=True, exist_ok=True)

            # MinerU creates a subdirectory structure: {output_dir}/{filename}/{method}/
            method_dir = persist_output_root / self.config["method"]
            content_file = method_dir / f"{stem}_content_list.json"
            
            if content_file.exists():
                with open(content_file) as f:
                    raw_content_list = json.load(f)
                # Persist assets (images) to stable location under working dir
                dest_root = config.get_working_dir() / "assets" / stem
                content_list = self._persist_assets(
                    content_list=raw_content_list,
                    source_root=method_dir,
                    dest_root=dest_root
                )
                return self._process_content_list(content_list)
            
            return []

    def _process_content_list(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and validate MinerU output"""
        processed_content = []
        
        for item in content_list:
            try:
                content_type = item.get("type", "text")
                
                if content_type == ContentType.TEXT:
                    processed_content.append(
                        TextContent(
                            text=item.get("text", ""),
                            text_level=item.get("text_level", 0),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
                elif content_type == ContentType.IMAGE and self.config["enable_images"]:
                    processed_content.append(
                        ImageContent(
                            img_path=item.get("img_path", ""),
                            image_caption=item.get("image_caption", []),
                            image_footnote=item.get("image_footnote", []),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
                elif content_type == ContentType.TABLE and self.config["enable_tables"]:
                    processed_content.append(
                        TableContent(
                            table_body=item.get("table_body", ""),
                            table_caption=item.get("table_caption", []),
                            table_footnote=item.get("table_footnote", []),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
                elif content_type == ContentType.EQUATION and self.config["enable_equations"]:
                    processed_content.append(
                        EquationContent(
                            latex=item.get("latex", ""),
                            text=item.get("text", ""),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
            except Exception as e:
                logger.warning(f"Failed to process content item: {str(e)}")
                continue
        
        return processed_content

class DoclingParser(BaseParser):
    async def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse document using Docling (fallback implementation)"""
        
        # For now, implement a basic fallback that extracts text
        # This can be enhanced later with actual Docling integration
        
        try:
            # Try to use docling if available
            with tempfile.TemporaryDirectory() as temp_dir:
                docling_cmd = str(Path(sys.executable).parent / "docling")
                cmd = [
                    docling_cmd,
                    "convert",
                    file_path,
                    "--output-dir", temp_dir,
                    "--output-format", "json"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Read the generated JSON file
                    output_file = Path(temp_dir) / f"{Path(file_path).stem}.json"
                    if output_file.exists():
                        with open(output_file) as f:
                            raw_content_list = json.load(f)
                        # Persist assets (images) to stable location
                        stem = Path(file_path).stem
                        dest_root = config.get_working_dir() / "assets" / stem
                        content_list = self._persist_assets(
                            content_list=raw_content_list,
                            source_root=Path(temp_dir),
                            dest_root=dest_root
                        )
                        return self._process_content_list(content_list)
                
                # Fallback: if docling fails, use a simple text extraction
                logger.warning("Docling not available, using fallback text extraction")
                return await self._fallback_text_extraction(file_path)
                
        except Exception as e:
            logger.warning(f"Docling parsing failed: {str(e)}, using fallback")
            return await self._fallback_text_extraction(file_path)
    
    async def _fallback_text_extraction(self, file_path: str) -> List[Dict[str, Any]]:
        """Fallback text extraction for when Docling is not available"""
        try:
            # Check file extension
            file_ext = Path(file_path).suffix.lower()

            if file_ext in ['.txt', '.md', '.text']:
                # Handle plain text files
                content_list = []
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()

                    if text_content.strip():
                        # Split into paragraphs
                        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]

                        for i, paragraph in enumerate(paragraphs):
                            # Determine text level based on markdown headers
                            text_level = 0
                            if paragraph.startswith('#'):
                                # Count # symbols
                                header_match = re.match(r'^(#+)', paragraph)
                                if header_match:
                                    text_level = len(header_match.group(1))
                                    paragraph = paragraph[text_level:].strip()

                            content_list.append({
                                "type": ContentType.TEXT,
                                "text": paragraph,
                                "page_idx": 0,  # Text files don't have pages
                                "text_level": text_level
                            })

                        return content_list
                    else:
                        return [{
                            "type": ContentType.TEXT,
                            "text": "Empty text file",
                            "page_idx": 0,
                            "text_level": 0
                        }]

            elif file_ext in ['.xlsx', '.xls']:
                # Handle Excel files with improved structure preservation
                return await self._parse_excel_file(file_path)

            else:
                # Try PDF extraction as fallback for other file types
                try:
                    import PyPDF2

                    content_list = []
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)

                        for page_num, page in enumerate(pdf_reader.pages):
                            text = page.extract_text()
                            if text.strip():
                                content_list.append({
                                    "type": ContentType.TEXT,
                                    "text": text.strip(),
                                    "page_idx": page_num,
                                    "text_level": 0
                                })

                    return content_list
                except Exception as e:
                    logger.warning(f"PDF extraction failed for {file_ext} file: {str(e)}")
                    return [{
                        "type": ContentType.TEXT,
                        "text": f"Unsupported file type {file_ext}: {Path(file_path).name}",
                        "page_idx": 0,
                        "text_level": 0
                    }]

        except ImportError:
            logger.error("PyPDF2 not available for fallback text extraction")
            # Return a basic text content indicating the file was processed
            return [{
                "type": ContentType.TEXT,
                "text": f"Document {Path(file_path).name} processed (text extraction not available)",
                "page_idx": 0,
                "text_level": 0
            }]
        except Exception as e:
            logger.error(f"Fallback text extraction failed: {str(e)}")
            # Return basic content on any error
            return [{
                "type": ContentType.TEXT,
                "text": f"Document {Path(file_path).name} could not be processed",
                "page_idx": 0,
                "text_level": 0
            }]
    
    async def _parse_excel_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Minimal Excel parsing fallback using pandas/openpyxl.
        Produces TABLE items by converting sheets to markdown-like text.
        """
        try:
            import pandas as pd
        except Exception as e:
            logger.warning(f"pandas not available for Excel parsing: {str(e)}")
            return [{
                "type": ContentType.TEXT,
                "text": f"Excel file {Path(file_path).name} processed (no table parser available)",
                "page_idx": 0,
                "text_level": 0
            }]

        try:
            # Read all sheets without dtype coercion, keep headers
            sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        except Exception as e:
            logger.warning(f"Failed reading Excel via pandas: {str(e)}")
            return [{
                "type": ContentType.TEXT,
                "text": f"Excel file {Path(file_path).name} could not be parsed",
                "page_idx": 0,
                "text_level": 0
            }]

        content_list: List[Dict[str, Any]] = []
        max_rows_per_chunk = 200

        for sheet_name, df in sheets.items():
            # Fill NaNs for readability
            safe_df = df.copy()
            safe_df = safe_df.where(pd.notnull(safe_df), "")

            total_rows = len(safe_df)
            if total_rows == 0:
                table_body = f"Sheet: {sheet_name}\n(Empty sheet)"
                content_list.append({
                    "type": ContentType.TABLE,
                    "table_body": table_body,
                    "table_caption": [f"Sheet {sheet_name}"],
                    "table_footnote": [],
                    "page_idx": 0
                })
                continue

            # Chunk by rows
            for start in range(0, total_rows, max_rows_per_chunk):
                end = min(start + max_rows_per_chunk, total_rows)
                chunk_df = safe_df.iloc[start:end]

                # Convert to simple pipe table (limited width)
                # Cap very long cell text to keep memory reasonable
                def truncate_cell(val: Any) -> str:
                    s = str(val)
                    return (s[:500] + "…") if len(s) > 500 else s

                limited_df = chunk_df.applymap(truncate_cell)
                # Include headers
                header = list(limited_df.columns)
                rows = [header] + limited_df.astype(str).values.tolist()

                # Build markdown-like table
                def row_to_pipe(r: List[str]) -> str:
                    return "| " + " | ".join(r) + " |"

                table_lines = []
                if header:
                    table_lines.append(row_to_pipe([str(h) for h in header]))
                    table_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                    for r in limited_df.astype(str).values.tolist():
                        table_lines.append(row_to_pipe([str(c) for c in r]))
                else:
                    # No headers
                    for r in rows:
                        table_lines.append(row_to_pipe([str(c) for c in r]))

                table_body = (
                    f"Sheet: {sheet_name} (rows {start + 1}-{end} of {total_rows})\n" +
                    "\n".join(table_lines)
                )

                content_list.append({
                    "type": ContentType.TABLE,
                    "table_body": table_body,
                    "table_caption": [f"Sheet {sheet_name}"],
                    "table_footnote": [],
                    "page_idx": 0
                })

        return content_list
    
    def _process_content_list(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and validate content list (same as MineruParser)"""
        processed_content = []
        
        for item in content_list:
            try:
                content_type = item.get("type", "text")
                
                if content_type == ContentType.TEXT:
                    processed_content.append(
                        TextContent(
                            text=item.get("text", ""),
                            text_level=item.get("text_level", 0),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
                elif content_type == ContentType.IMAGE and self.config["enable_images"]:
                    processed_content.append(
                        ImageContent(
                            img_path=item.get("img_path", ""),
                            image_caption=item.get("image_caption", []),
                            image_footnote=item.get("image_footnote", []),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
                elif content_type == ContentType.TABLE and self.config["enable_tables"]:
                    processed_content.append(
                        TableContent(
                            table_body=item.get("table_body", ""),
                            table_caption=item.get("table_caption", []),
                            table_footnote=item.get("table_footnote", []),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
                elif content_type == ContentType.EQUATION and self.config["enable_equations"]:
                    processed_content.append(
                        EquationContent(
                            equation=item.get("equation", ""),
                            page_idx=item.get("page_idx", 0)
                        ).dict()
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to process content item: {str(e)}")
                continue
        
        return processed_content

class ParserFactory:
    @staticmethod
    def get_parser(parser_type: Optional[str] = None) -> BaseParser:
        """Get appropriate parser based on type or config"""
        parser_type = parser_type or config.PARSER

        if parser_type.lower() == "mineru":
            return MineruParser()
        elif parser_type.lower() == "docling":
            return DoclingParser()
        else:
            raise ValueError(f"Unsupported parser type: {parser_type}")

    @staticmethod
    async def parse_document(file_path: str, parser_type: Optional[str] = None, ingest_summary: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Parse document using appropriate parser"""
        from pathlib import Path

        # Auto-select parser based on file type (following RAG-Anything template)
        file_ext = Path(file_path).suffix.lower()

        # For Office documents (Excel, Word, PowerPoint), prefer MinerU
        # MinerU handles Office documents natively with full content extraction
        if file_ext in ['.xlsx', '.xls', '.docx', '.doc', '.pptx', '.ppt']:
            if not parser_type or parser_type.lower() == "auto":
                parser_type = "mineru"  # MinerU is better for Office docs
        elif file_ext in ['.pdf']:
            # Use MinerU as default for PDF parsing - it provides superior layout analysis
            if not parser_type or parser_type.lower() == "auto":
                parser_type = "mineru"
        else:
            # For other formats, use configured default
            pass

        # Get the appropriate parser
        parser = ParserFactory.get_parser(parser_type)
        actual_parser_used = parser.__class__.__name__.replace('Parser', '').lower()

        # Track parser usage in summary if provided
        if ingest_summary is not None:
            ingest_summary['parser_used'] = actual_parser_used

        # Try the primary parser first
        try:
            result = await parser.parse_document(file_path)
            if result:  # If we got meaningful content, return it
                return result
        except Exception as e:
            logger.warning(f"Primary parser {parser.__class__.__name__} failed for {file_path}: {str(e)}")
            # Track error in summary
            if ingest_summary is not None:
                if 'errors' not in ingest_summary:
                    ingest_summary['errors'] = {}
                error_key = f"Primary parser {actual_parser_used} failed"
                ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1

        # Fallback: try alternative parser if primary fails
        if isinstance(parser, MineruParser):
            try:
                fallback_parser = DoclingParser()
                result = await fallback_parser.parse_document(file_path)
                if result:
                    logger.info(f"Using Docling parser as fallback for {file_path}")
                    # Update parser used in summary
                    if ingest_summary is not None:
                        ingest_summary['parser_used'] = 'docling (fallback)'
                    return result
            except Exception as e:
                logger.warning(f"Fallback parser also failed for {file_path}: {str(e)}")
                # Track error in summary
                if ingest_summary is not None:
                    if 'errors' not in ingest_summary:
                        ingest_summary['errors'] = {}
                    error_key = "Fallback parser docling failed"
                    ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1
        elif isinstance(parser, DoclingParser):
            try:
                fallback_parser = MineruParser()
                result = await fallback_parser.parse_document(file_path)
                if result:
                    logger.info(f"Using MinerU parser as fallback for {file_path}")
                    # Update parser used in summary
                    if ingest_summary is not None:
                        ingest_summary['parser_used'] = 'mineru (fallback)'
                    return result
            except Exception as e:
                logger.warning(f"Fallback parser also failed for {file_path}: {str(e)}")
                # Track error in summary
                if ingest_summary is not None:
                    if 'errors' not in ingest_summary:
                        ingest_summary['errors'] = {}
                    error_key = "Fallback parser mineru failed"
                    ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1

        # Final fallback: basic text extraction
        logger.warning(f"All parsers failed for {file_path}, using basic fallback")
        fallback_parser = DoclingParser()
        # Track fallback usage in summary
        if ingest_summary is not None:
            ingest_summary['parser_used'] = 'fallback text extraction'
            if 'warnings' not in ingest_summary:
                ingest_summary['warnings'] = {}
            warning_key = "All parsers failed, using basic text extraction"
            ingest_summary['warnings'][warning_key] = ingest_summary['warnings'].get(warning_key, 0) + 1
        return await fallback_parser._fallback_text_extraction(file_path)