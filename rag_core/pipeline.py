from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import shutil, json
from fastapi import UploadFile
from rag_core.config import AppConfig
from rag_core.schemas import IngestOptions, IngestResponse, QueryRequest, QueryResponse, StatusResponse, DocumentListResponse
from rag_core.parsers import BaseParser, excel_native_summary
from rag_core.storage import StorageBundle
from rag_core.utils import cache_key, content_doc_id, template_chunk, calculate_file_hash
from rag_core.processors import ModalProcessors
from rag_core.llm_bedrock import BedrockLLM

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
    def __init__(self, cfg: AppConfig, stores: StorageBundle, parser: BaseParser, procs: ModalProcessors, llm: BedrockLLM | None = None):
        self.cfg = cfg
        self.stores = stores
        self.parser = parser
        self.procs = procs
        self.llm = llm

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
                    total_blocks=existing_info["total_blocks"],
                    by_type=existing_info["by_type"],
                    notes="duplicate_skipped",
                    is_duplicate=True,
                    duplicate_doc_id=existing_doc_id
                )

        key = cache_key(
            dest, self.cfg.parser, opts.parse_method or self.cfg.parse_method,
            lang=opts.lang, start_page=opts.start_page, end_page=opts.end_page,
            formula=opts.enable_formula, table=opts.enable_table, backend=opts.backend, source=opts.source
        )
        cached = await self.stores.cache.get(key)
        if cached:
            return IngestResponse(
                doc_id=cached["doc_id"], file_name=file.filename,
                total_blocks=len(cached["content_list"]),
                by_type=cached["by_type"], notes="cache_hit"
            )

        ext = dest.suffix.lower()
        content_list: List[Dict[str,Any]] = []
        if ext == ".pdf":
            content_list = self.parser.parse_pdf(dest)
        elif ext in [".jpg",".jpeg",".png",".bmp",".tiff",".tif",".gif",".webp"]:
            content_list = self.parser.parse_image(dest)
        elif ext in [".doc",".docx",".ppt",".pptx",".xls",".xlsx",".html",".htm",".xhtml"]:
            content_list = self.parser.parse_office_doc(dest)
            if opts.excel_native_summary and ext in [".xls",".xlsx"]:
                content_list += excel_native_summary(dest)
        else:
            content_list = self.parser.parse_document(dest)

        if not content_list:
            raise RuntimeError("Parsing produced no content")

        doc_id = content_doc_id(content_list)

        by_type = {}
        chunk_payloads = {}
        chunk_order = 0
        for item in content_list:
            t = item.get("type","text")
            by_type[t] = by_type.get(t,0) + 1
            desc = await self.procs.describe_item(item)
            text = template_chunk(item, desc)
            chunk_id = f"chunk-{doc_id}-{chunk_order}"
            chunk_order += 1
            chunk_payloads[chunk_id] = {
                "text": text,
                "doc_id": doc_id,
                "type": t,
                "page_idx": item.get("page_idx")
            }

        await self.stores.text_chunks.upsert(chunk_payloads)
        await self.stores.vectors.upsert(chunk_payloads)

        await self.stores.cache.put(key, {
            "doc_id": doc_id,
            "content_list": content_list,
            "by_type": by_type
        })

        # Register the document to prevent future duplicates
        self.stores.documents.register_document(
            doc_id=doc_id,
            filename=file.filename,
            file_hash=file_hash,
            file_size=file_size,
            content_list=content_list,
            by_type=by_type
        )

        return IngestResponse(doc_id=doc_id, file_name=file.filename,
                              total_blocks=sum(by_type.values()), by_type=by_type)

    async def query(self, req: QueryRequest) -> QueryResponse:
        k = req.k
        hits = await self.stores.vectors.search(req.query, k=k)

        # Fallback (no LLM)
        if not self.llm:
            ctx = "\n\n".join([f"[{h['chunk_id']}] {h.get('text_preview','')}" for h in hits])
            answer = f"Top-{k} evidence:\n\n{ctx[:4000]}"
            return QueryResponse(answer=answer, hits=hits, used_mode=req.mode)

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

        # Call LLM, but never let None/Type errors bubble up
# inside RagPipeline.query (replace the try/except block)

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
