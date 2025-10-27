from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import time
import base64
from pathlib import Path
from .config import config
from .llm_unified import UnifiedLLM
from .storage import StorageManager
from .multimodal import MultimodalProcessor
from .utils import VectorIndex, ChunkManager, DocumentRegistry
from .schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        self.config = config
        self.llm = UnifiedLLM()
        self.storage = StorageManager()
        self.multimodal = MultimodalProcessor()
        
        # Initialize local storage
        working_dir = config.get_working_dir()
        self.vector_index = VectorIndex(working_dir / "vectors")
        self.chunk_manager = ChunkManager(working_dir / "text_chunks")
        self.doc_registry = DocumentRegistry(working_dir / "kv/document_registry")
    
    async def process_query(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        """Process query request"""
        
        start_time = time.time()
        
        try:
            # Enhance query if multimodal content provided
            if request.multimodal_content:
                enhanced_query = await self._enhance_query(
                    request.query,
                    request.multimodal_content
                )
            else:
                enhanced_query = request.query
            
            # Get query embedding
            embeddings = await self.llm.get_embeddings(
                texts=[enhanced_query]
            )
            query_embedding = embeddings[0]
            
            # Search based on query type
            if request.query_type == "text":
                result = await self._process_text_query(
                    enhanced_query,
                    query_embedding,
                    request.mode
                )
            elif request.query_type == "multimodal":
                result = await self._process_multimodal_query(
                    enhanced_query,
                    query_embedding,
                    request.multimodal_content,
                    request.mode
                )
            elif request.query_type == "vlm_enhanced":
                result = await self._process_vlm_query(
                    enhanced_query,
                    query_embedding,
                    request.mode
                )
            else:
                raise ValueError(f"Unsupported query type: {request.query_type}")
            
            # Get processing time
            processing_time = time.time() - start_time
            
            # Create response
            return QueryResponse(
                result=result["answer"],
                query_type=request.query_type,
                processing_time=processing_time,
                entities_found=result.get("entities", []),
                multimodal_context=result.get("multimodal_context", [])
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise
    
    async def _enhance_query(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """Enhance query with multimodal context"""
        
        enhanced_parts = [f"User query: {query}"]
        
        for content in multimodal_content:
            content_type = content.get("type", "unknown")
            
            if content_type == "image":
                description = await self._describe_image_for_query(content)
                enhanced_parts.append(f"\nRelated image: {description}")
            elif content_type == "table":
                description = await self._describe_table_for_query(content)
                enhanced_parts.append(f"\nRelated table: {description}")
            elif content_type == "equation":
                description = await self._describe_equation_for_query(content)
                enhanced_parts.append(f"\nRelated equation: {description}")
        
        enhanced_parts.append("\nPlease provide a comprehensive answer considering all the provided information.")
        
        return "\n".join(enhanced_parts)
    
    async def _describe_image_for_query(
        self,
        content: Dict[str, Any]
    ) -> str:
        """Generate image description for query context"""
        
        image_path = content.get("img_path")
        captions = content.get("image_caption", [])
        
        if image_path and Path(image_path).exists():
            # Use vision model
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            prompt = "Describe the main content and key elements in this image for query context."
            
            description = await self.llm.analyze_image(
                image_data=image_data,
                prompt=prompt
            )
            
            return description
        else:
            # Use existing metadata
            parts = []
            if image_path:
                parts.append(f"Image: {image_path}")
            if captions:
                parts.append(f"Caption: {', '.join(captions)}")
            
            return "; ".join(parts) if parts else "Image content unavailable"
    
    async def _describe_table_for_query(
        self,
        content: Dict[str, Any]
    ) -> str:
        """Generate table description for query context"""
        
        table_body = content.get("table_body", "")
        captions = content.get("table_caption", [])
        
        prompt = f"""
        Summarize this table data concisely:
        
        Table:
        {table_body}
        
        Caption: {', '.join(captions) if captions else 'None'}
        """
        
        return await self.llm.generate_text(prompt)
    
    async def _describe_equation_for_query(
        self,
        content: Dict[str, Any]
    ) -> str:
        """Generate equation description for query context"""
        
        latex = content.get("latex", "")
        text = content.get("text", "")
        
        prompt = f"""
        Explain this equation concisely:
        
        LaTeX: {latex}
        Description: {text}
        """
        
        return await self.llm.generate_text(prompt)
    
    async def _process_text_query(
        self,
        query: str,
        query_embedding: List[float],
        mode: str
    ) -> Dict[str, Any]:
        """Process text query with graph-enhanced retrieval"""

        # Extract entities from query first
        query_entities = await self._extract_entities_from_query(query)

        # Get chunks and entities from vector search
        vector_results = self.vector_index.search(
            query_vector=query_embedding,
            limit=5,
            threshold=0.0  # Lower threshold to get more results
        )

        chunks = []
        entities_found = []

        # Get chunks and their related entities
        for result in vector_results:
            chunk_id = result["metadata"]["id"]
            chunk = self.chunk_manager.get_chunk(chunk_id)
            if chunk:
                chunks.append({
                    "content": chunk["content"],
                    "score": result["score"],
                    "chunk_id": chunk_id
                })

                # Get entities related to this chunk from graph
                chunk_entities = await self._get_entities_for_chunk(chunk_id)
                entities_found.extend(chunk_entities)

        # If query entities found, use graph traversal for better context
        if query_entities and self.storage.graph.driver:
            graph_context = await self._get_graph_context(query_entities)
            if graph_context:
                # Add graph context to chunks
                chunks.append({
                    "content": f"Related entities and context: {graph_context}",
                    "score": 0.9,
                    "chunk_id": "graph_context"
                })

        # Build context
        context = "\n\n".join([
            f"[Score: {chunk['score']:.2f}]\n{chunk['content']}"
            for chunk in chunks
        ])

        # Generate answer
        prompt = f"""
        Answer the following question based on the provided context:

        Question: {query}

        Context:
        {context}

        Provide a comprehensive answer using only the information from the context.
        If the context doesn't contain enough information, say so.
        """

        answer = await self.llm.generate_text(
            prompt=prompt,
            system_prompt="You are a helpful assistant that provides accurate answers based on the given context."
        )

        return {
            "answer": answer,
            "chunks": chunks,
            "entities": list(set(entities_found))  # Remove duplicates
        }
    
    async def _process_multimodal_query(
        self,
        query: str,
        query_embedding: List[float],
        multimodal_content: List[Dict[str, Any]],
        mode: str
    ) -> Dict[str, Any]:
        """Process multimodal query"""
        
        # Get text context
        text_results = await self._process_text_query(
            query, query_embedding, mode
        )
        
        # Process multimodal content
        multimodal_context = []
        for content in multimodal_content:
            try:
                # Get context for the item
                context = await self._get_multimodal_context(content)
                
                # Process with appropriate processor
                chunk_content, entity_info = await self.multimodal.process_item(
                    content, context, "query"
                )
                
                multimodal_context.append({
                    "content": chunk_content,
                    "entity": entity_info
                })
            except Exception as e:
                logger.warning(f"Failed to process multimodal content: {str(e)}")
                continue
        
        # Build combined context
        combined_context = text_results["chunks"]
        for item in multimodal_context:
            combined_context.append({
                "content": item["content"],
                "score": 1.0  # Explicitly provided content
            })
        
        # Generate answer
        context_text = "\n\n".join([
            f"[Score: {chunk['score']:.2f}]\n{chunk['content']}"
            for chunk in combined_context
        ])
        
        prompt = f"""
        Answer the following question considering both text and multimodal content:
        
        Question: {query}
        
        Context:
        {context_text}
        
        Provide a comprehensive answer that integrates information from both text and multimodal content.
        """
        
        answer = await self.llm.generate_text(
            prompt=prompt,
            system_prompt="You are a helpful assistant that provides accurate answers based on both text and multimodal content."
        )
        
        return {
            "answer": answer,
            "chunks": text_results["chunks"],
            "multimodal_context": [item["content"] for item in multimodal_context]
        }
    
    async def _process_vlm_query(
        self,
        query: str,
        query_embedding: List[float],
        mode: str
    ) -> Dict[str, Any]:
        """Process query with vision-language model"""
        
        # Get initial results
        results = await self._process_text_query(
            query, query_embedding, mode
        )
        
        # Extract image references
        image_paths = []
        for chunk in results["chunks"]:
            content = chunk["content"]
            if "img_path" in content:
                # Simple extraction - you might want more sophisticated parsing
                for line in content.split("\n"):
                    if line.startswith("Path: "):
                        image_path = line.replace("Path: ", "").strip()
                        if Path(image_path).exists():
                            image_paths.append(image_path)
        
        if not image_paths:
            return results
        
        # Build multimodal messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can understand both text and images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        Question: {query}
                        
                        Context:
                        {results['chunks'][0]['content']}
                        
                        Please analyze the provided images and answer the question.
                        """
                    }
                ]
            }
        ]
        
        # Add images
        for image_path in image_paths[:4]:  # Limit to 4 images
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })
        
        # Get VLM response
        response = await self.llm.analyze_image(
            image_data="",  # Not used with messages
            prompt="",  # Not used with messages
            messages=messages
        )
        
        return {
            "answer": response,
            "chunks": results["chunks"],
            "multimodal_context": image_paths
        }
    
    async def _get_multimodal_context(
        self,
        content: Dict[str, Any]
    ) -> str:
        """Get context for multimodal content"""
        
        # Get entity context from graph
        entity_name = content.get("entity_name")
        if entity_name:
            context = await self.storage.get_entity_context(entity_name)
            if context:
                return json.dumps(context, indent=2)
        
        # Fallback to basic context
        return json.dumps(content, indent=2)

    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query text"""
        if not self.storage.graph.driver:
            return []

        try:
            # Use LLM to extract entities from query
            prompt = f"""
            Extract key entities (people, organizations, locations, products, concepts, technical terms) from this query:

            Query: {query}

            Return a JSON array of entity names:
            ["Entity1", "Entity2", "Entity3"]
            """

            response = await self.llm.generate_text(
                prompt=prompt,
                system_prompt="You are an expert at extracting key terms and entities from text."
            )

            # Parse entities from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                import json
                entities = json.loads(json_match.group())
                return entities
            else:
                return []

        except Exception as e:
            logger.warning(f"Failed to extract entities from query: {e}")
            return []

    async def _get_entities_for_chunk(self, chunk_id: str) -> List[str]:
        """Get entities related to a chunk from the graph"""
        if not self.storage.graph.driver:
            return []

        try:
            query = """
            MATCH (e:Entity)-[r]-(c:Chunk {chunk_id: $chunk_id})
            RETURN e.name, e.entity_type
            """

            with self.storage.graph.driver.session() as session:
                result = session.run(query, {"chunk_id": chunk_id})
                entities = []
                for record in result:
                    entities.append(f"{record['e.name']} ({record['e.entity_type']})")
                return entities

        except Exception as e:
            logger.warning(f"Failed to get entities for chunk {chunk_id}: {e}")
            return []

    async def _get_graph_context(self, query_entities: List[str]) -> str:
        """Get context from graph based on query entities"""
        if not self.storage.graph.driver or not query_entities:
            return ""

        try:
            # First, check what relationship types actually exist in the database
            schema_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN collect(relationshipType) as relationship_types
            """

            available_relationships = []
            with self.storage.graph.driver.session() as session:
                try:
                    schema_result = session.run(schema_query)
                    for record in schema_result:
                        available_relationships = record['relationship_types']
                        break
                except Exception:
                    # Fallback: try common relationship types
                    available_relationships = ['REFERENCES', 'APPEARS_ON']

                # Find entities that match or are related to query entities
                entity_names = "', '".join(query_entities)

                # Build dynamic relationship type filter based on what's available
                if available_relationships:
                    rel_types = '|'.join(available_relationships)
                    query = f"""
                    MATCH (e:Entity)
                    WHERE e.name IN ['{entity_names}']
                    OPTIONAL MATCH (e)-[r:{rel_types}*1..2]-(related:Entity)
                    RETURN e.name, e.entity_type, e.description,
                           collect(DISTINCT related.name + ' (' + related.entity_type + ')') as related_entities
                    """
                else:
                    # Fallback query without relationships if no schema info available
                    query = f"""
                    MATCH (e:Entity)
                    WHERE e.name IN ['{entity_names}']
                    RETURN e.name, e.entity_type, e.description, [] as related_entities
                    """

                result = session.run(query)
                context_parts = []

                for record in result:
                    main_entity = f"{record['e.name']} ({record['e.entity_type']}): {record['e.description']}"
                    context_parts.append(main_entity)

                    if record['related_entities']:
                        related = ", ".join(record['related_entities'])
                        context_parts.append(f"Related to: {related}")

                return "\n".join(context_parts) if context_parts else ""

        except Exception as e:
            logger.warning(f"Failed to get graph context: {e}")
            return ""
