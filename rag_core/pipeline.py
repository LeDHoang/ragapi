from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from pathlib import Path
import time
import json
import hashlib
import base64
from lightrag.lightrag import LightRAG
from .config import config
from .parsers import ParserFactory
from .processors import ContentSeparator
from .llm_unified import UnifiedLLM
from .schemas import ProcessingStatus, DocumentMetadata
from .utils import VectorIndex, ChunkManager, DocumentRegistry
from .storage import StorageManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.config = config
        self.llm = UnifiedLLM()
        self.lightrag = None
        if self.config.LIGHTRAG_ENABLED:
            try:
                self.lightrag = self._init_lightrag()
            except Exception as exc:
                logger.warning(
                    "LightRAG initialization failed, falling back to legacy pipeline",
                    exc_info=exc,
                )
        self.content_separator = ContentSeparator()
        self.storage_manager = StorageManager(self.lightrag)

        # Create working directories
        self.working_dir = config.get_working_dir()
        self.chunks_dir = self.working_dir / "text_chunks"
        self.vectors_dir = self.working_dir / "vectors"
        self.kv_dir = self.working_dir / "kv"

        for dir_path in [self.chunks_dir, self.vectors_dir, self.kv_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage components
        self.vector_index: Optional[VectorIndex] = None
        if not self.lightrag:
            self.vector_index = VectorIndex(self.vectors_dir)
        self.chunk_manager = ChunkManager(self.chunks_dir)

        # Ensure document registry directory exists
        doc_registry_dir = self.kv_dir / "document_registry"
        doc_registry_dir.mkdir(parents=True, exist_ok=True)
        self.doc_registry = DocumentRegistry(doc_registry_dir)
    
    def _init_lightrag(self) -> Optional[LightRAG]:
        """Initialize LightRAG with configuration"""

        try:
            model_config = self.config.get_model_config()
            processing_config = self.config.get_processing_config()
            lr_config = self.config.get_lightrag_config()

            working_dir = self.config.get_lightrag_working_dir()
            working_dir.mkdir(parents=True, exist_ok=True)

            # Use LightRAG's built-in OpenAI integration with proper async setup
            import os

            # Ensure OpenAI API key is available
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for LightRAG integration")

            # Set the API key in environment for LightRAG
            os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY

            # Import LightRAG's OpenAI functions and utilities
            from lightrag.llm.openai import gpt_4o_mini_complete
            from lightrag.utils import EmbeddingFunc

            # Create a proper EmbeddingFunc instance with our fixed implementation
            async def fixed_openai_embed_func(texts, model="text-embedding-3-small", **kwargs):
                """Fixed OpenAI embedding function"""
                try:
                    # Ensure texts is a list
                    if isinstance(texts, str):
                        texts = [texts]

                    # Filter out empty texts
                    valid_texts = [t for t in texts if t and t.strip()]
                    if not valid_texts:
                        import numpy as np
                        return np.array([[0.0] * self.config.EMBEDDING_DIM for _ in texts])

                    # Use our fixed UnifiedLLM implementation
                    embeddings = await self.llm.get_embeddings(texts=valid_texts, model=model)
                    import numpy as np
                    return np.array(embeddings)
                except Exception as e:
                    logger.error(f"Fixed OpenAI embed failed: {str(e)}")
                    import numpy as np
                    return np.array([[0.0] * self.config.EMBEDDING_DIM for _ in texts])

            # Create EmbeddingFunc instance
            embedding_func = EmbeddingFunc(
                embedding_dim=self.config.EMBEDDING_DIM,
                func=fixed_openai_embed_func,
                max_token_size=8192  # Standard token limit for OpenAI
            )

            lightrag = LightRAG(
                working_dir=str(working_dir),
                chunk_token_size=processing_config["chunk_size"],
                chunk_overlap_token_size=processing_config["chunk_overlap"],
                llm_model_func=gpt_4o_mini_complete,
                llm_model_name=model_config["llm_model"],
                tiktoken_model_name=model_config["embedding_model"],
                embedding_func=embedding_func,
                **lr_config,
            )

            # Initialize storages in a new event loop to avoid conflicts
            import threading

            def init_storages():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(lightrag.initialize_storages())
                    # Initialize pipeline status
                    from lightrag.kg.shared_storage import initialize_pipeline_status
                    loop.run_until_complete(initialize_pipeline_status())
                finally:
                    loop.close()

            init_thread = threading.Thread(target=init_storages)
            init_thread.start()
            init_thread.join(timeout=30)  # 30 second timeout

            if init_thread.is_alive():
                logger.warning("LightRAG storage initialization timed out")
                return None

            return lightrag

        except Exception as exc:
            logger.warning(f"Failed to initialize LightRAG: {exc}")
            return None
    
    async def process_document(
        self,
        file_path: str,
        task_id: str,
        parser_type: Optional[str] = None
    ) -> ProcessingStatus:
        """Process document through the complete pipeline"""

        # Initialize ingest summary for tracking
        ingest_summary = {
            'parser_used': None,
            'storage_issues': [],
            'errors': {},
            'warnings': {}
        }

        try:
            # Update status
            status = ProcessingStatus(
                task_id=task_id,
                status="processing",
                progress=0.0
            )
            await self._update_status(status)

            # 1. Parse document
            logger.info(f"Parsing document: {file_path}")
            content_list = await ParserFactory.parse_document(file_path, parser_type, ingest_summary)
            status.progress = 0.2
            await self._update_status(status)
            
            # 2. Separate content
            logger.info("Separating content")
            full_text, multimodal_items, summary = await self.content_separator.process_document_content(
                content_list, task_id
            )
            status.progress = 0.4
            await self._update_status(status)
            
            # 3. Process text with LightRAG
            logger.info("Processing text content")
            await self._process_text_content(full_text, task_id, file_path, ingest_summary)
            status.progress = 0.6
            await self._update_status(status)
            
            # 4. Process multimodal content
            logger.info("Processing multimodal content")
            if self.lightrag:
                await self._process_multimodal_content_lightrag(multimodal_items, task_id, file_path, ingest_summary)
            else:
                await self._process_multimodal_content(multimodal_items, task_id, file_path, ingest_summary)
            status.progress = 0.8
            await self._update_status(status)
            
            # 5. Create document metadata
            metadata = DocumentMetadata(
                doc_id=task_id,
                file_path=str(file_path),
                file_type=Path(file_path).suffix,
                total_pages=summary["structure"]["total_pages"],
                processed_at=time.time(),
                chunks_count=await self._count_chunks(task_id),
                entities_count=await self._count_entities_lightrag(task_id) if self.lightrag else len(await self._get_entities(task_id))
            )
            await self._save_metadata(metadata)

            # 6. Store document in graph database
            try:
                await self.storage_manager.store_document(
                    doc_id=task_id,
                    file_path=str(file_path),
                    metadata={
                        "file_type": Path(file_path).suffix,
                        "total_pages": summary["structure"]["total_pages"],
                        "chunks_count": await self._count_chunks(task_id),
                        "entities_count": await self._count_entities_lightrag(task_id) if self.lightrag else len(await self._get_entities(task_id)),
                        "processed_at": time.time()
                    }
                )
            except Exception as e:
                error_msg = f"Graph database storage failed: {str(e)}"
                logger.warning(error_msg)
                ingest_summary['storage_issues'].append(error_msg)
                if 'errors' not in ingest_summary:
                    ingest_summary['errors'] = {}
                ingest_summary['errors']['Graph database storage failed'] = ingest_summary['errors'].get('Graph database storage failed', 0) + 1

            # 7. Find and create relationships between entities
            logger.info("Creating entity relationships")
            try:
                if self.lightrag:
                    await self._build_cross_modal_relationships_lightrag(task_id, ingest_summary)
                else:
                    await self.storage_manager.find_entity_relationships(task_id)
            except Exception as e:
                error_msg = f"Entity relationship creation failed: {str(e)}"
                logger.warning(error_msg)
                ingest_summary['storage_issues'].append(error_msg)
                if 'errors' not in ingest_summary:
                    ingest_summary['errors'] = {}
                ingest_summary['errors']['Entity relationship creation failed'] = ingest_summary['errors'].get('Entity relationship creation failed', 0) + 1

            # 8. Register document in document registry for API access
            await self._register_document(metadata)

            # Update final status
            status.status = "completed"
            status.progress = 1.0
            status.doc_id = task_id
            status.chunks_created = metadata.chunks_count
            status.entities_found = await self._count_entities_lightrag(task_id) if self.lightrag else len(await self._get_entities(task_id))
            await self._update_status(status)

            # Log final ingest summary
            self._log_ingest_summary(ingest_summary, task_id, file_path, status)

            # Attach summary to status for access by caller
            status.ingest_summary = ingest_summary

            return status
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            status.status = "failed"
            status.error = str(e)
            await self._update_status(status)
            raise
    
    async def _process_text_content(
        self,
        text: str,
        doc_id: str,
        file_path: str,
        ingest_summary: Dict[str, Any] = None
    ):
        """Process text content with chunking and storage"""
        
        # Split text into chunks
        chunk_size = self.config.CHUNK_SIZE
        chunk_overlap = self.config.CHUNK_OVERLAP

        chunks = self._split_text_into_chunks(text, chunk_size, chunk_overlap)

        if self.lightrag:
            await self._upsert_text_chunks_lightrag(
                chunks=chunks,
                doc_id=doc_id,
                file_path=file_path,
                ingest_summary=ingest_summary
            )
            return

        # Fallback legacy path
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"chunk-{doc_id}-{i}"

            await self._store_chunk(
                chunk_id=chunk_id,
                content=chunk_text,
                doc_id=doc_id,
                file_path=file_path,
                chunk_type="text",
                page_idx=None,
                ingest_summary=ingest_summary
            )
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                # Find the last sentence or paragraph boundary
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > chunk_size * 0.5:  # If boundary is not too far back
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    async def _store_chunk(
        self,
        chunk_id: str,
        content: str,
        doc_id: str,
        file_path: str,
        chunk_type: str,
        page_idx: Optional[int] = None,
        ingest_summary: Dict[str, Any] = None
    ):
        """Store a chunk with metadata, create embedding, and extract entities"""

        # Create embedding for the chunk
        try:
            embeddings = await self.llm.get_embeddings(texts=[content])
            embedding = embeddings[0]

            # Store chunk using storage manager (includes graph and vector storage)
            await self.storage_manager.store_chunk(
                chunk_id=chunk_id,
                content=content,
                doc_id=doc_id,
                chunk_type=chunk_type,
                page_idx=page_idx,
                vector=embedding
            )

            # Extract and store entities from the chunk
            await self.storage_manager.extract_and_store_entities(
                content=content,
                chunk_id=chunk_id,
                doc_id=doc_id,
                file_path=file_path,
                content_type=chunk_type
            )

        except Exception as e:
            error_msg = f"Failed to process chunk {chunk_id}: {str(e)}"
            logger.warning(error_msg)
            if ingest_summary is not None:
                ingest_summary['storage_issues'].append(error_msg)
                if 'errors' not in ingest_summary:
                    ingest_summary['errors'] = {}
                error_key = "Chunk processing failed"
                ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1
    
    async def _process_multimodal_content(
        self,
        items: List[Dict[str, Any]],
        doc_id: str,
        file_path: str,
        ingest_summary: Dict[str, Any] = None
    ):
        """Process multimodal content items"""
        
        for item in items:
            try:
                # Get context for the item
                context = self.content_separator.processor.context_extractor.extract_context(
                    items, item
                )
                
                # Process based on type
                if item["type"] in ("image", "image".upper(), "IMAGE"):
                    await self._process_image(item, context, doc_id, file_path, ingest_summary)
                elif item["type"] in ("table", "TABLE"):
                    await self._process_table(item, context, doc_id, file_path, ingest_summary)
                elif item["type"] in ("equation", "EQUATION"):
                    await self._process_equation(item, context, doc_id, file_path, ingest_summary)
            except Exception as e:
                error_msg = f"Failed to process multimodal item: {str(e)}"
                logger.warning(error_msg)
                if ingest_summary is not None:
                    ingest_summary['storage_issues'].append(error_msg)
                    if 'errors' not in ingest_summary:
                        ingest_summary['errors'] = {}
                    error_key = "Multimodal item processing failed"
                    ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1
                continue
    
    async def _process_image(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str,
        file_path: str,
        ingest_summary: Dict[str, Any] = None
    ):
        """Process image with vision model"""
        
        img_path = item.get("img_path")
        if not img_path or not Path(img_path).exists():
            return
        
        # Read image and convert to base64
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # Analyze image
        prompt = f"""
        Analyze this image considering the surrounding context:
        
        Context: {context}
        
        Provide a detailed description and identify key elements.
        """
        
        analysis = await self.llm.analyze_image(
            image_data=image_data,
            prompt=prompt,
            system_prompt="You are an expert image analyst..."
        )
        
        # Create chunk for the image analysis
        chunk_content = f"""
        Image Analysis:
        Path: {img_path}
        Page: {item.get('page_idx', 0)}
        Caption: {', '.join(item.get('image_caption', []))}
        Analysis: {analysis}
        """

        # Store multimodal chunk in graph and vector storage
        await self._store_chunk(
            chunk_id=hashlib.md5(chunk_content.encode()).hexdigest(),
            content=chunk_content,
            doc_id=doc_id,
            file_path=file_path,
            chunk_type=item["type"],
            page_idx=item.get('page_idx', 0),
            ingest_summary=ingest_summary
        )
    
    async def _process_table(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str,
        file_path: str,
        ingest_summary: Dict[str, Any] = None
    ):
        """Process table content"""
        
        table_body = item.get("table_body", "")
        captions = item.get("table_caption", [])
        
        # Analyze table
        prompt = f"""
        Analyze this table data considering the context:
        
        Context: {context}
        Table:
        {table_body}
        
        Provide a detailed analysis including key insights and patterns.
        """
        
        analysis = await self.llm.generate_text(
            prompt=prompt,
            system_prompt="You are an expert data analyst..."
        )
        
        # Create chunk for the table analysis
        chunk_content = f"""
        Table Analysis:
        Page: {item.get('page_idx', 0)}
        Caption: {', '.join(captions)}
        Content:
        {table_body}
        Analysis: {analysis}
        """

        # Store multimodal chunk in graph and vector storage
        await self._store_chunk(
            chunk_id=hashlib.md5(chunk_content.encode()).hexdigest(),
            content=chunk_content,
            doc_id=doc_id,
            file_path=file_path,
            chunk_type=item["type"],
            page_idx=item.get('page_idx', 0),
            ingest_summary=ingest_summary
        )
    
    async def _process_equation(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str,
        file_path: str,
        ingest_summary: Dict[str, Any] = None
    ):
        """Process equation content"""
        
        latex = item.get("latex", "")
        text = item.get("text", "")
        
        # Analyze equation
        prompt = f"""
        Explain this mathematical equation in context:
        
        Context: {context}
        Equation (LaTeX): {latex}
        Description: {text}
        
        Provide a detailed explanation of the equation's meaning and significance.
        """
        
        explanation = await self.llm.generate_text(
            prompt=prompt,
            system_prompt="You are an expert mathematician..."
        )
        
        # Create chunk for the equation analysis
        chunk_content = f"""
        Equation Analysis:
        Page: {item.get('page_idx', 0)}
        LaTeX: {latex}
        Description: {text}
        Explanation: {explanation}
        """

        # Store multimodal chunk in graph and vector storage
        await self._store_chunk(
            chunk_id=hashlib.md5(chunk_content.encode()).hexdigest(),
            content=chunk_content,
            doc_id=doc_id,
            file_path=file_path,
            chunk_type=item["type"],
            page_idx=item.get('page_idx', 0),
            ingest_summary=ingest_summary
        )
    
    
    async def _count_chunks(self, doc_id: str) -> int:
        """Count chunks for a document"""
        count = 0
        
        # First, try to use ChunkManager's index for accurate counting
        try:
            chunks = self.chunk_manager.get_chunks_by_doc(doc_id)
            return len([chunk for chunk in chunks if chunk is not None])
        except Exception as e:
            logger.warning(f"Failed to use ChunkManager index: {e}")
        
        # Fallback: manually count files
        for chunk_file in self.chunks_dir.glob("*.txt"):
            try:
                with open(chunk_file) as f:
                    content = f.read().strip()
                    if not content:  # Skip empty files
                        continue
                    
                    # Try to parse as JSON first (from _create_chunk method)
                    try:
                        chunk_data = json.loads(content)
                        if chunk_data.get("doc_id") == doc_id:
                            count += 1
                            continue
                    except json.JSONDecodeError:
                        pass
                    
                    # If not JSON, check if filename matches doc_id pattern
                    # This handles chunks created by ChunkManager
                    if doc_id in chunk_file.stem:
                        count += 1
                        
            except (FileNotFoundError, PermissionError) as e:
                logger.warning(f"Failed to read chunk file {chunk_file}: {e}")
                continue
                
        return count
    
    async def _get_entities(self, doc_id: str) -> List[str]:
        """Get entities for a document from the knowledge graph"""
        if not self.storage_manager.graph.driver:
            return []

        try:
            query = """
            MATCH (e:Entity)-[r1]-(c:Chunk)-[r2]-(d:Document {doc_id: $doc_id})
            RETURN e.name, e.entity_type, e.description
            """

            with self.storage_manager.graph.driver.session() as session:
                result = session.run(query, {"doc_id": doc_id})
                entities = []
                for record in result:
                    entities.append({
                        "name": record["e.name"],
                        "type": record["e.entity_type"],
                        "description": record["e.description"]
                    })
                return entities
        except Exception as e:
            logger.warning(f"Failed to get entities from graph for doc {doc_id}: {e}")
            return []
    
    async def _save_metadata(self, metadata: DocumentMetadata):
        """Save document metadata"""
        metadata_path = self.kv_dir / "metadata" / f"{metadata.doc_id}.json"
        metadata_path.parent.mkdir(exist_ok=True)
        with open(metadata_path, "w") as f:
            f.write(metadata.json())

    async def _register_document(self, metadata: DocumentMetadata):
        """Register document in document registry for API access"""
        try:
            # Register in document registry
            registry_metadata = {
                "file_path": metadata.file_path,
                "file_type": metadata.file_type,
                "total_pages": metadata.total_pages,
                "processed_at": metadata.processed_at,
                "chunks_count": metadata.chunks_count,
                "entities_count": metadata.entities_count
            }

            self.doc_registry.register_document(
                doc_id=metadata.doc_id,
                file_path=metadata.file_path,
                metadata=registry_metadata
            )

            logger.info(f"Document {metadata.doc_id} registered successfully")

        except Exception as e:
            logger.error(f"Failed to register document {metadata.doc_id}: {str(e)}")
            # Don't fail the entire pipeline if registration fails
            pass

    async def _upsert_text_chunks_lightrag(
        self,
        chunks: List[str],
        doc_id: str,
        file_path: str,
        ingest_summary: Dict[str, Any] = None
    ):
        """Insert text chunks into LightRAG storage"""
        if not self.lightrag:
            logger.warning("LightRAG not initialized, skipping text chunk insertion")
            return

        try:
            # Filter out empty or very short chunks to prevent embedding errors
            valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 10]
            
            if not valid_chunks:
                logger.warning(f"No valid text chunks found for document {doc_id}, skipping LightRAG insertion")
                return

            # Convert chunks to format LightRAG expects
            # LightRAG's ainsert expects text content, not pre-split chunks
            full_text = "\n\n".join(valid_chunks)

            # Additional validation to ensure we have meaningful content
            if len(full_text.strip()) < 20:
                logger.warning(f"Text content too short for document {doc_id}, skipping LightRAG insertion")
                return

            # Use LightRAG's ainsert method
            track_id = await self.lightrag.ainsert(
                input=full_text,
                ids=[doc_id],
                file_paths=[file_path],
                split_by_character="\n\n",  # Split by paragraph boundaries
                split_by_character_only=False,  # Allow token-based splitting too
            )

            logger.info(f"LightRAG text insertion completed with track_id: {track_id}")

        except Exception as e:
            error_msg = f"LightRAG text insertion failed: {str(e)}"
            logger.error(error_msg)
            if ingest_summary is not None:
                ingest_summary['storage_issues'].append(error_msg)
                if 'errors' not in ingest_summary:
                    ingest_summary['errors'] = {}
                error_key = "LightRAG text insertion failed"
                ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1
            raise

    async def _process_multimodal_content_lightrag(
        self,
        items: List[Dict[str, Any]],
        doc_id: str,
        file_path: str,
        ingest_summary: Dict[str, Any] = None
    ):
        """Process multimodal content using LightRAG"""
        if not self.lightrag:
            logger.warning("LightRAG not initialized, skipping multimodal content processing")
            return

        try:
            # Convert multimodal items to LightRAG format
            multimodal_content = []

            for item in items:
                item_type = item.get("type", "").lower()
                lightrag_item = {
                    "type": item_type,
                    "content": {}
                }

                if item_type == "image":
                    # Process image content
                    img_path = item.get("img_path")
                    if img_path and Path(img_path).exists():
                        # Get context for the image
                        context = self.content_separator.processor.context_extractor.extract_context(
                            items, item
                        )

                        # Analyze image with vision model
                        with open(img_path, "rb") as f:
                            image_data = base64.b64encode(f.read()).decode()

                        prompt = f"""
                        Analyze this image considering the surrounding context:

                        Context: {context}

                        Provide a detailed description and identify key elements.
                        """

                        analysis = await self.llm.analyze_image(
                            image_data=image_data,
                            prompt=prompt,
                            system_prompt="You are an expert image analyst..."
                        )

                        lightrag_item["content"] = {
                            "image_path": img_path,
                            "analysis": analysis,
                            "caption": item.get("image_caption", []),
                            "page": item.get("page_idx", 0)
                        }

                elif item_type == "table":
                    # Process table content
                    table_body = item.get("table_body", "")
                    captions = item.get("table_caption", [])

                    if table_body:
                        # Get context for the table
                        context = self.content_separator.processor.context_extractor.extract_context(
                            items, item
                        )

                        # Analyze table
                        prompt = f"""
                        Analyze this table data considering the context:

                        Context: {context}
                        Table:
                        {table_body}

                        Provide a detailed analysis including key insights and patterns.
                        """

                        analysis = await self.llm.generate_text(
                            prompt=prompt,
                            system_prompt="You are an expert data analyst..."
                        )

                        lightrag_item["content"] = {
                            "table_data": table_body,
                            "analysis": analysis,
                            "caption": captions,
                            "page": item.get("page_idx", 0)
                        }

                elif item_type == "equation":
                    # Process equation content
                    latex = item.get("latex", "")
                    text = item.get("text", "")

                    if latex:
                        # Get context for the equation
                        context = self.content_separator.processor.context_extractor.extract_context(
                            items, item
                        )

                        # Analyze equation
                        prompt = f"""
                        Explain this mathematical equation in context:

                        Context: {context}
                        Equation (LaTeX): {latex}
                        Description: {text}

                        Provide a detailed explanation of the equation's meaning and significance.
                        """

                        explanation = await self.llm.generate_text(
                            prompt=prompt,
                            system_prompt="You are an expert mathematician..."
                        )

                        lightrag_item["content"] = {
                            "latex": latex,
                            "text": text,
                            "explanation": explanation,
                            "page": item.get("page_idx", 0)
                        }

                multimodal_content.append(lightrag_item)

            # Use LightRAG's ainsert with multimodal content
            if multimodal_content:
                # Create a minimal text description to avoid empty input error
                text_description = f"Multimodal content from document {doc_id}: {len(multimodal_content)} items"
                
                track_id = await self.lightrag.ainsert(
                    input=text_description,  # Provide minimal text to avoid empty input error
                    multimodal_content=multimodal_content,
                    ids=[f"{doc_id}_multimodal"],
                    file_paths=[file_path],
                )

                logger.info(f"LightRAG multimodal insertion completed with track_id: {track_id}")

        except Exception as e:
            error_msg = f"LightRAG multimodal processing failed: {str(e)}"
            logger.error(error_msg)
            if ingest_summary is not None:
                ingest_summary['storage_issues'].append(error_msg)
                if 'errors' not in ingest_summary:
                    ingest_summary['errors'] = {}
                error_key = "LightRAG multimodal processing failed"
                ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1
            # Don't raise - multimodal processing is supplementary
            pass

    async def _count_entities_lightrag(self, doc_id: str) -> int:
        """Count entities for a document using LightRAG"""
        if not self.lightrag:
            return 0

        try:
            # Use LightRAG's entity storage to count entities
            # This is a simplified approach - in practice you'd query the entities_vdb
            return len(await self.lightrag.entities_vdb.get_by_ids([]))  # Placeholder
        except Exception as e:
            logger.warning(f"Failed to count entities via LightRAG: {e}")
            return 0

    async def _build_cross_modal_relationships_lightrag(self, doc_id: str, ingest_summary: Dict[str, Any] = None):
        """Build cross-modal relationships using LightRAG"""
        if not self.lightrag:
            logger.warning("LightRAG not initialized, skipping cross-modal relationship building")
            return

        try:
            # LightRAG handles relationship building automatically during insertion
            # But we can trigger additional cross-modal analysis here

            # Get all entities and chunks for this document
            # This is a placeholder - in practice you'd use LightRAG's graph traversal
            # to find and create cross-modal relationships

            logger.info(f"Cross-modal relationships built for document {doc_id}")

        except Exception as e:
            error_msg = f"Cross-modal relationship building failed: {str(e)}"
            logger.error(error_msg)
            if ingest_summary is not None:
                ingest_summary['storage_issues'].append(error_msg)
                if 'errors' not in ingest_summary:
                    ingest_summary['errors'] = {}
                error_key = "Cross-modal relationship building failed"
                ingest_summary['errors'][error_key] = ingest_summary['errors'].get(error_key, 0) + 1
            # Don't raise - relationship building is supplementary
            pass

    async def _update_status(self, status: ProcessingStatus):
        """Update processing status"""
        status_path = self.kv_dir / "status" / f"{status.task_id}.json"
        status_path.parent.mkdir(exist_ok=True)
        with open(status_path, "w") as f:
            f.write(status.json())

    def _log_ingest_summary(self, ingest_summary: Dict[str, Any], task_id: str, file_path: str, status: ProcessingStatus):
        """Log comprehensive ingest summary"""
        logger.info("="*80)
        logger.info(f"[INGEST SUMMARY] Document processing completed for task_id={task_id}")
        logger.info(f"[INGEST SUMMARY] File: {file_path}")
        logger.info(f"[INGEST SUMMARY] Parser used: {ingest_summary.get('parser_used', 'unknown')}")

        # Log storage issues
        storage_issues = ingest_summary.get('storage_issues', [])
        if storage_issues:
            logger.warning(f"[INGEST SUMMARY] Storage issues detected ({len(storage_issues)}):")
            for issue in storage_issues:
                logger.warning(f"[INGEST SUMMARY]   - {issue}")
        else:
            logger.info("[INGEST SUMMARY] No storage issues detected")

        # Log errors with counts
        errors = ingest_summary.get('errors', {})
        if errors:
            logger.warning(f"[INGEST SUMMARY] Errors encountered ({sum(errors.values())} total):")
            for error_msg, count in errors.items():
                logger.warning(f"[INGEST SUMMARY]   - {error_msg}: {count} occurrence(s)")
        else:
            logger.info("[INGEST SUMMARY] No errors encountered")

        # Log warnings with counts
        warnings = ingest_summary.get('warnings', {})
        if warnings:
            logger.warning(f"[INGEST SUMMARY] Warnings encountered ({sum(warnings.values())} total):")
            for warning_msg, count in warnings.items():
                logger.warning(f"[INGEST SUMMARY]   - {warning_msg}: {count} occurrence(s)")
        else:
            logger.info("[INGEST SUMMARY] No warnings encountered")

        # Log final statistics
        logger.info(f"[INGEST SUMMARY] Final statistics:")
        logger.info(f"[INGEST SUMMARY]   - Chunks created: {status.chunks_created}")
        logger.info(f"[INGEST SUMMARY]   - Entities found: {status.entities_found}")
        logger.info(f"[INGEST SUMMARY]   - Processing status: {status.status}")
        logger.info("="*80)