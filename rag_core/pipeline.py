from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from pathlib import Path
import time
import json
import hashlib
import base64
# from lightrag import LightRAG  # TODO: Fix LightRAG import - not available in current version
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
        # self.lightrag = self._init_lightrag()  # TODO: Fix LightRAG initialization
        self.content_separator = ContentSeparator()
        self.storage_manager = StorageManager()

        # Create working directories
        self.working_dir = config.get_working_dir()
        self.chunks_dir = self.working_dir / "text_chunks"
        self.vectors_dir = self.working_dir / "vectors"
        self.kv_dir = self.working_dir / "kv"

        for dir_path in [self.chunks_dir, self.vectors_dir, self.kv_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage components
        self.vector_index = VectorIndex(self.vectors_dir)
        self.chunk_manager = ChunkManager(self.chunks_dir)
        self.doc_registry = DocumentRegistry(self.kv_dir / "document_registry")
    
    # def _init_lightrag(self) -> LightRAG:
    #     """Initialize LightRAG with configuration"""
    #     
    #     model_config = self.config.get_model_config()
    #     processing_config = self.config.get_processing_config()
    #     
    #     return LightRAG(
    #         working_dir=str(self.working_dir),
    #         embedding_model=model_config["embedding_model"],
    #         embedding_dim=model_config["embedding_dim"],
    #         chunk_size=processing_config["chunk_size"],
    #         chunk_overlap=processing_config["chunk_overlap"]
    #     )
    
    async def process_document(
        self,
        file_path: str,
        task_id: str,
        parser_type: Optional[str] = None
    ) -> ProcessingStatus:
        """Process document through the complete pipeline"""
        
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
            content_list = await ParserFactory.parse_document(file_path, parser_type)
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
            await self._process_text_content(full_text, task_id, file_path)
            status.progress = 0.6
            await self._update_status(status)
            
            # 4. Process multimodal content
            logger.info("Processing multimodal content")
            await self._process_multimodal_content(multimodal_items, task_id, file_path)
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
                entities_count=len(await self._get_entities(task_id))
            )
            await self._save_metadata(metadata)

            # 6. Store document in graph database
            await self.storage_manager.store_document(
                doc_id=task_id,
                file_path=str(file_path),
                metadata={
                    "file_type": Path(file_path).suffix,
                    "total_pages": summary["structure"]["total_pages"],
                    "chunks_count": await self._count_chunks(task_id),
                    "entities_count": len(await self._get_entities(task_id)),
                    "processed_at": time.time()
                }
            )

            # 7. Find and create relationships between entities
            logger.info("Creating entity relationships")
            await self.storage_manager.find_entity_relationships(task_id)

            # 8. Register document in document registry for API access
            await self._register_document(metadata)

            # Update final status
            status.status = "completed"
            status.progress = 1.0
            status.doc_id = task_id
            status.chunks_created = metadata.chunks_count
            status.entities_found = len(await self._get_entities(task_id))
            await self._update_status(status)
            
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
        file_path: str
    ):
        """Process text content with chunking and storage"""
        
        # Split text into chunks
        chunk_size = self.config.CHUNK_SIZE
        chunk_overlap = self.config.CHUNK_OVERLAP
        
        chunks = self._split_text_into_chunks(text, chunk_size, chunk_overlap)
        
        # Process each chunk
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"chunk-{doc_id}-{i}"
            
            # Store chunk
            await self._store_chunk(
                chunk_id=chunk_id,
                content=chunk_text,
                doc_id=doc_id,
                file_path=file_path,
                chunk_type="text",
                page_idx=None
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
        page_idx: Optional[int] = None
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
            logger.warning(f"Failed to process chunk {chunk_id}: {e}")
    
    async def _process_multimodal_content(
        self,
        items: List[Dict[str, Any]],
        doc_id: str,
        file_path: str
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
                    await self._process_image(item, context, doc_id, file_path)
                elif item["type"] in ("table", "TABLE"):
                    await self._process_table(item, context, doc_id, file_path)
                elif item["type"] in ("equation", "EQUATION"):
                    await self._process_equation(item, context, doc_id, file_path)
            except Exception as e:
                logger.warning(f"Failed to process multimodal item: {str(e)}")
                continue
    
    async def _process_image(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str,
        file_path: str
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
            page_idx=item.get('page_idx', 0)
        )
    
    async def _process_table(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str,
        file_path: str
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
            page_idx=item.get('page_idx', 0)
        )
    
    async def _process_equation(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str,
        file_path: str
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
            page_idx=item.get('page_idx', 0)
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
    
    async def _update_status(self, status: ProcessingStatus):
        """Update processing status"""
        status_path = self.kv_dir / "status" / f"{status.task_id}.json"
        status_path.parent.mkdir(exist_ok=True)
        with open(status_path, "w") as f:
            f.write(status.json())