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
            # Build MinerU command (try CLI first, then python -m mineru)
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
            
            # Try different ways to run mineru
            venv_mineru = str(Path(sys.executable).parent / "mineru")
            tried_cmds = [
                ["mineru", *base_args],  # Direct command
                [sys.executable, "-m", "mineru", *base_args],  # Python module
                [venv_mineru, *base_args],  # venv/bin/mineru
            ]
            
            last_err: Optional[str] = None
            for cmd in tried_cmds:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        break
                    last_err = result.stderr
                except FileNotFoundError as e:
                    last_err = str(e)
                    continue
            else:
                # None succeeded
                raise Exception(f"MinerU failed to run: {last_err}")
            
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
                cmd = [
                    "docling",
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

            else:
                # Try PDF extraction as fallback
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
    async def parse_document(file_path: str, parser_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse document using appropriate parser"""
        parser = ParserFactory.get_parser(parser_type)
        return await parser.parse_document(file_path)