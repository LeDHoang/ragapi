from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import shutil, json
from datetime import datetime
from fastapi import UploadFile
from rag_core.config import AppConfig
from rag_core.schemas import IngestOptions, IngestResponse, QueryRequest, QueryResponse, StatusResponse, DocumentListResponse
from rag_core.parsers import BaseParser, excel_native_summary
from rag_core.storage import StorageBundle
from rag_core.utils import cache_key_with_overlay, content_doc_id, template_chunk, calculate_file_hash
from rag_core.processors import ModalProcessors
from rag_core.llm_unified import UnifiedLLM, create_llm
from rag_core.overlay import LayoutOverlay, create_overlay_generator

ANSWER_SYSTEM = (
    "You are a retrieval-augmented assistant. Write a clear, readable answer using the supplied chunks. "
    "Prefer bullet points and short paragraphs. Only use facts from the chunks. "
    "Cite evidence with bracketed chunk ids like [#chunk-XYZ] inline."
)

def _format_cited_context(hits: List[Dict[str,Any]], text_dir: Path, max_chars_per_chunk: int = 1200) -> str:
    sections = []
    for h in hits:
        cid = h["chunk_id"]
        page = h.get("page_idx")
        p = text_dir / f"{cid}.txt"
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            raw = h.get("text_preview","") or ""
        snippet = raw[:max_chars_per_chunk]
        header = f"[{cid}] (type={h.get('type')}, page={page})"
        sections.append(f"{header}\n{snippet}")
    return "\n\n".join(sections)

class RagPipeline:
    def __init__(self, cfg: AppConfig, stores: StorageBundle, parser: BaseParser, procs: ModalProcessors, llm: UnifiedLLM | None = None):
        self.cfg = cfg
        self.stores = stores
        self.parser = parser
        self.procs = procs
        self.llm = llm
        self.overlay_generator = create_overlay_generator(cfg) if cfg.export_layout_overlay else None

    async def ingest_file(self, file: UploadFile, opts: IngestOptions) -> IngestResponse:
        dest = Path(self.cfg.output_dir) / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # Calculate file hash for duplicate detection
        file_hash = calculate_file_hash(dest)
        file_size = dest.stat().st_size

        # Check for duplicates before processing
        is_duplicate, existing_doc_id = self.stores.documents.is_duplicate(
            file.filename, file_hash, file_size
        )

        if is_duplicate and existing_doc_id:
            # Return existing document info without reprocessing
            existing_info = self.stores.documents.get_document_info(existing_doc_id)
            if existing_info:
                return IngestResponse(
                    doc_id=existing_doc_id,
                    file_name=file.filename,
                    file_size=file_size,
                    total_blocks=existing_info["total_blocks"],
                    by_type=existing_info["by_type"],
                    processed_at=existing_info.get("processed_at", datetime.now()),
                    notes="duplicate_skipped",
                    is_duplicate=True,
                    duplicate_doc_id=existing_doc_id
                )

        key = cache_key_with_overlay(
            dest, self.cfg.parser, opts.parse_method or self.cfg.parse_method,
            lang=opts.lang, start_page=opts.start_page, end_page=opts.end_page,
            formula=opts.enable_formula, table=opts.enable_table,
            image=opts.enable_image, backend=opts.backend, source=opts.source,
            export_overlay=opts.export_layout_overlay
        )
        cached = await self.stores.cache.get(key)
        if cached:
            return IngestResponse(
                doc_id=cached["doc_id"], 
                file_name=file.filename,
                file_size=file_size,
                total_blocks=len(cached["content_list"]),
                by_type=cached["by_type"], 
                processed_at=datetime.now(),
                notes="cache_hit"
            )

        # Parse document content
        content_list = await self._parse_document_content(dest, opts)

        if not content_list:
            raise RuntimeError("Parsing produced no content")

        doc_id = content_doc_id(content_list)

        # Filter content based on options
        content_list = self._filter_content_by_options(content_list, opts)

        # Build context map for multimodal processing
        page_context_map = self.procs.context_extractor.build_page_context_map(content_list) if self.procs.context_extractor else {}

        # Process content in batches for memory efficiency
        by_type = {}
        chunk_payloads = {}
        chunk_order = 0

        # Process items in batches
        batch_size = 50  # Process 50 items at a time
        for i in range(0, len(content_list), batch_size):
            batch_items = content_list[i:i + batch_size]

            for item in batch_items:
                t = item.get("type", "text")
                by_type[t] = by_type.get(t, 0) + 1

                # Get context for multimodal items
                context = None
                if self.procs.context_extractor and t in ["image", "table", "equation"]:
                    context = self.procs.context_extractor.extract_context(
                        content_list, item
                    )

                # Generate description with context
                desc = await self.procs.describe_item(item, context)

                # Create chunk
                text = template_chunk(item, desc)
                chunk_id = f"chunk-{doc_id}-{chunk_order}"
                chunk_order += 1

                chunk_payloads[chunk_id] = {
                    "text": text,
                    "doc_id": doc_id,
                    "type": t,
                    "page_idx": item.get("page_idx"),
                    "bbox": item.get("bbox"),
                    "page_size": item.get("page_size")
                }

        # Upsert chunks and vectors
        await self.stores.text_chunks.upsert(chunk_payloads)
        await self.stores.vectors.upsert(chunk_payloads)

        # Cache the results
        await self.stores.cache.put(key, {
            "doc_id": doc_id,
            "content_list": content_list,
            "by_type": by_type
        })

        # Register the document
        self.stores.documents.register_document(
            doc_id=doc_id,
            filename=file.filename,
            file_hash=file_hash,
            file_size=file_size,
            content_list=content_list,
            by_type=by_type
        )

        # Generate layout overlay if requested
        notes_parts = []
        if (opts.export_layout_overlay or self.cfg.export_layout_overlay) and dest.suffix.lower() == ".pdf":
            try:
                overlay_result = self.overlay_generator.render(dest, content_list)
                notes_parts.append(f"Layout overlay generated: {overlay_result['total_pages']} pages, {overlay_result['total_elements']} elements")
            except Exception as e:
                notes_parts.append(f"Layout overlay failed: {str(e)}")

        notes = "; ".join(notes_parts) if notes_parts else None

        return IngestResponse(
            doc_id=doc_id,
            file_name=file.filename,
            file_size=file_size,
            total_blocks=sum(by_type.values()),
            by_type=by_type,
            processed_at=datetime.now(),
            notes=notes
        )

    async def _parse_document_content(self, dest: Path, opts: IngestOptions) -> List[Dict[str, Any]]:
        """Parse document content with chunking support for large files"""
        ext = dest.suffix.lower()
        content_list: List[Dict[str, Any]] = []

        # Handle different file types
        if ext == ".pdf":
            content_list = await self._parse_pdf_with_chunking(dest, opts)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"]:
            content_list = self.parser.parse_image(dest)
        elif ext in [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".html", ".htm", ".xhtml"]:
            content_list = self.parser.parse_office_doc(dest)
            if opts.excel_native_summary and ext in [".xls", ".xlsx"]:
                content_list += excel_native_summary(dest)
        else:
            content_list = self.parser.parse_document(dest)

        return content_list

    async def _parse_pdf_with_chunking(self, pdf_path: Path, opts: IngestOptions) -> List[Dict[str, Any]]:
        """Parse PDF with optional chunking for large files"""
        # Check if chunking is enabled and needed
        if not self.cfg.parse_chunk_size_pages or self.cfg.parse_chunk_size_pages <= 0:
            return self.parser.parse_pdf(pdf_path)

        # For now, implement simple chunking - in a full implementation,
        # you'd want to use page ranges or split the PDF
        # This is a placeholder that calls the regular parser
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            doc.close()

            if total_pages <= self.cfg.parse_chunk_size_pages:
                return self.parser.parse_pdf(pdf_path)

            # For large PDFs, we'd implement chunking here
            # For now, fall back to regular parsing with a warning
            print(f"Large PDF detected ({total_pages} pages). Consider implementing chunked parsing.")

        except ImportError:
            pass  # Fall back to regular parsing

        return self.parser.parse_pdf(pdf_path)

    def _filter_content_by_options(self, content_list: List[Dict[str, Any]], opts: IngestOptions) -> List[Dict[str, Any]]:
        """Filter content based on processing options"""
        filtered = []

        for item in content_list:
            item_type = item.get("type", "text")

            # Apply filters based on options
            if item_type == "table" and not opts.enable_table:
                continue
            elif item_type == "image" and not opts.enable_image:
                continue
            elif item_type == "equation" and not opts.enable_formula:
                continue

            filtered.append(item)

        return filtered

    async def query(self, req: QueryRequest) -> QueryResponse:
        k = req.k or req.max_hits  # Use k if provided, otherwise fall back to max_hits
        hits = await self.stores.vectors.search(req.query, k=k)

        # Fallback (no LLM)
        if not self.llm:
            ctx = "\n\n".join([f"[{h['chunk_id']}] {h.get('text_preview','')}" for h in hits])
            answer = f"Top-{k} evidence:\n\n{ctx[:4000]}"
            return QueryResponse(answer=answer, hits=hits, used_mode=req.mode)

        # Handle multimodal queries
        if req.mode in ["multimodal", "vlm_enhanced"]:
            return await self._handle_multimodal_query(req, hits)
        else:
            return await self._handle_text_query(req, hits)

    async def _handle_text_query(self, req: QueryRequest, hits: List[Dict[str, Any]]) -> QueryResponse:
        """Handle standard text-based queries"""
        # Build prompt
        text_dir = Path(self.stores.text_chunks.path)
        context_block = _format_cited_context(hits, text_dir)

        user_prompt = (
            "Question:\n"
            f"{req.query}\n\n"
            "Evidence chunks (use only these; cite with [#chunk-id] inline):\n"
            f"{context_block}\n\n"
            "Write a concise, readable answer:\n"
            "- Bullet points for key facts\n"
            "- Short paragraphs (2–4 sentences)\n"
            "- Include units/dates/numbers when present\n"
            "- Strictly avoid speculation; if unknown, say so\n"
            "- Add inline citations like [#chunk-...] next to claims\n"
        )

        try:
            answer = self.llm.generate(ANSWER_SYSTEM, user_prompt, temperature=0.2) or ""
            return QueryResponse(answer=answer, hits=hits, used_mode=req.mode)
        except Exception as e:
            # Show the real LLM error so you can fix creds / model / region quickly
            bullets = []
            for h in hits:
                cid = h["chunk_id"]
                prev = h.get("text_preview","") or ""
                bullets.append(f"- {prev.strip()[:300]} [#{cid}]")
            answer = (
                "LLM error: " + str(e) + "\n\n"
                "Summary based on retrieved chunks (no generation):\n" + "\n".join(bullets)
            )
            return QueryResponse(answer=answer, hits=hits, used_mode=req.mode)

    async def _handle_multimodal_query(self, req: QueryRequest, hits: List[Dict[str, Any]]) -> QueryResponse:
        """Handle multimodal queries with enhanced context"""
        # Filter hits to include multimodal content
        multimodal_hits = [h for h in hits if h.get("type") in ["image", "table", "equation"]]

        if not multimodal_hits:
            # Fallback to text query if no multimodal content found
            return await self._handle_text_query(req, hits)

        # Build enhanced prompt with multimodal context
        text_dir = Path(self.stores.text_chunks.path)
        context_parts = []

        # Add text context
        context_parts.append("Text context:")
        context_parts.append(_format_cited_context(hits, text_dir))

        # Add multimodal context descriptions
        multimodal_context = []
        for hit in multimodal_hits[:3]:  # Limit to top 3 multimodal items
            chunk_path = text_dir / f"{hit['chunk_id']}.txt"
            if chunk_path.exists():
                content = chunk_path.read_text(encoding="utf-8")
                multimodal_context.append(f"[{hit['chunk_id']}] {content[:500]}...")

        if multimodal_context:
            context_parts.append("Multimodal content:")
            context_parts.append("\n".join(multimodal_context))

        context_block = "\n\n".join(context_parts)

        user_prompt = (
            "Question:\n"
            f"{req.query}\n\n"
            "Available context (use only these; cite with [#chunk-id] inline):\n"
            f"{context_block}\n\n"
            "This query involves multimodal content (images, tables, equations). "
            "Provide a comprehensive answer that:\n"
            "- Explains visual elements if images are present\n"
            "- Summarizes key data if tables are present\n"
            "- Explains mathematical concepts if equations are present\n"
            "- Integrates all available information into a cohesive response\n"
            "- Uses inline citations like [#chunk-...] for all claims\n"
        )

        try:
            answer = self.llm.generate(ANSWER_SYSTEM, user_prompt, temperature=0.2) or ""
            return QueryResponse(answer=answer, hits=hits, used_mode=req.mode)
        except Exception as e:
            # Fallback to text query
            return await self._handle_text_query(req, hits)

    async def status(self) -> StatusResponse:
        text_dir = Path(self.stores.text_chunks.path)
        chunks = len(list(text_dir.glob("*.txt")))
        kv_dir = Path(self.stores.cache.path)
        docs = len(list(kv_dir.glob("*.json")))
        return StatusResponse(docs_indexed=docs, chunks_indexed=chunks)

    async def list_documents(self) -> DocumentListResponse:
        """List all processed documents"""
        documents = self.stores.documents.list_documents()
        return DocumentListResponse(documents=documents, total=len(documents))

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its associated data"""
        # Delete from document registry
        registry_deleted = self.stores.documents.delete_document(doc_id)

        # Delete associated chunks from text store
        text_dir = Path(self.stores.text_chunks.path)
        chunks_deleted = 0
        for chunk_file in text_dir.glob(f"chunk-{doc_id}-*.txt"):
            chunk_file.unlink()
            chunks_deleted += 1

        # Delete from cache if exists
        cache_deleted = False
        kv_dir = Path(self.stores.cache.path)
        for cache_file in kv_dir.glob("*.json"):
            try:
                cached_data = json.loads(cache_file.read_text())
                if cached_data.get("doc_id") == doc_id:
                    cache_file.unlink()
                    cache_deleted = True
                    break
            except (json.JSONDecodeError, IOError):
                continue

        # TODO: Delete from vector store (would need more sophisticated implementation)
        # For now, we'll leave vectors but they won't be returned in searches without chunks

        return registry_deleted or chunks_deleted > 0 or cache_deleted
