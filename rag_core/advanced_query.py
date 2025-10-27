"""
Advanced Query Module using LightRAG

This module provides enhanced query capabilities using LightRAG's graph traversal
and semantic search features, while maintaining compatibility with the existing API.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import time
from pathlib import Path

from lightrag.lightrag import QueryParam
from .config import config
from .llm_unified import UnifiedLLM
from .storage import StorageManager
from .schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


class AdvancedQueryProcessor:
    """Advanced query processor using LightRAG's capabilities"""

    def __init__(self, lightrag=None):
        self.lightrag = lightrag
        self.config = config
        self.llm = UnifiedLLM()
        self.storage = StorageManager(lightrag)

        # Initialize fallback storage for compatibility
        if not lightrag:
            from .utils import VectorIndex, ChunkManager, DocumentRegistry
            working_dir = config.get_working_dir()
            self.vector_index = VectorIndex(working_dir / "vectors")
            self.chunk_manager = ChunkManager(working_dir / "text_chunks")
            self.doc_registry = DocumentRegistry(working_dir / "kv/document_registry")

    async def process_query_lightrag(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        """Process query using LightRAG's advanced capabilities"""
        if not self.lightrag:
            logger.warning("LightRAG not available, falling back to legacy query processing")
            from .query import QueryProcessor
            fallback_processor = QueryProcessor()
            return await fallback_processor.process_query(request)

        start_time = time.time()

        try:
            # Enhance query if multimodal content provided
            enhanced_query = request.query
            if request.multimodal_content:
                enhanced_query = await self._enhance_query_lightrag(
                    request.query,
                    request.multimodal_content
                )

            # Use LightRAG's advanced query capabilities
            query_param = QueryParam(
                mode=request.mode,
                only_need_context=False,
                only_need_prompt=False,
                response_type="Multiple Paragraphs",
                stream=False,
                top_k=10,
                chunk_top_k=20,
                max_entity_tokens=6000,
                max_relation_tokens=8000,
                max_total_tokens=30000,
                enable_rerank=True,
                include_references=True,
            )

            # Execute query using LightRAG
            result = await self.lightrag.aquery(enhanced_query, param=query_param)

            # Extract entities and context from LightRAG's response
            entities_found = await self._extract_entities_from_lightrag_result(result)
            multimodal_context = []

            processing_time = time.time() - start_time

            return QueryResponse(
                result=result,
                query_type=request.query_type,
                processing_time=processing_time,
                entities_found=entities_found,
                multimodal_context=multimodal_context
            )

        except Exception as e:
            logger.error(f"LightRAG query processing failed: {str(e)}")
            # Fallback to legacy processing
            from .query import QueryProcessor
            fallback_processor = QueryProcessor()
            return await fallback_processor.process_query(request)

    async def _enhance_query_lightrag(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """Enhance query using LightRAG's multimodal understanding"""
        if not multimodal_content:
            return query

        enhanced_parts = [f"User query: {query}"]

        for content in multimodal_content:
            content_type = content.get("type", "unknown")

            if content_type == "image":
                description = await self._describe_image_lightrag(content)
                enhanced_parts.append(f"\nRelated image: {description}")
            elif content_type == "table":
                description = await self._describe_table_lightrag(content)
                enhanced_parts.append(f"\nRelated table: {description}")
            elif content_type == "equation":
                description = await self._describe_equation_lightrag(content)
                enhanced_parts.append(f"\nRelated equation: {description}")

        enhanced_parts.append(
            "\nPlease provide a comprehensive answer considering all the provided information."
        )

        return "\n".join(enhanced_parts)

    async def _describe_image_lightrag(
        self,
        content: Dict[str, Any]
    ) -> str:
        """Generate image description using LightRAG context"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", [])

        if image_path and Path(image_path).exists():
            # Use LightRAG's multimodal capabilities if available
            return f"Image: {image_path}, Caption: {', '.join(captions)}"
        else:
            return "; ".join(captions) if captions else "Image content unavailable"

    async def _describe_table_lightrag(
        self,
        content: Dict[str, Any]
    ) -> str:
        """Generate table description using LightRAG context"""
        table_body = content.get("table_body", "")
        captions = content.get("table_caption", [])

        return f"Table: {table_body[:200]}... Caption: {', '.join(captions)}" if captions else f"Table: {table_body[:200]}..."

    async def _describe_equation_lightrag(
        self,
        content: Dict[str, Any]
    ) -> str:
        """Generate equation description using LightRAG context"""
        latex = content.get("latex", "")
        text = content.get("text", "")

        return f"Equation: {latex}, Description: {text}"

    async def _extract_entities_from_lightrag_result(self, result: str) -> List[str]:
        """Extract entities from LightRAG's query result"""
        # This is a simplified extraction - in practice you'd parse LightRAG's structured response
        # For now, return empty list as LightRAG handles entity extraction internally
        return []

    async def multi_hop_traversal(
        self,
        start_entity: str,
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform multi-hop graph traversal using LightRAG"""
        if not self.lightrag:
            return {"error": "LightRAG not available"}

        try:
            # Use LightRAG's graph traversal capabilities
            # This would use LightRAG's graph storage to find paths between entities

            # For now, return a placeholder response
            return {
                "start_entity": start_entity,
                "max_hops": max_hops,
                "relationship_types": relationship_types,
                "paths": [],  # Would contain actual paths from LightRAG's graph
                "entities_found": [],
                "relationships_found": []
            }

        except Exception as e:
            logger.error(f"Multi-hop traversal failed: {str(e)}")
            return {"error": str(e)}

    async def semantic_similarity_search(
        self,
        query: str,
        limit: int = 10,
        entity_type: Optional[str] = None,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search using LightRAG's VDBs"""
        if not self.lightrag:
            # Fallback to legacy search
            embeddings = await self.llm.get_embeddings(texts=[query])
            return await self.storage.search_similar_entities(
                embeddings[0], entity_type, limit
            )

        try:
            # Get query embedding
            embeddings = await self.llm.get_embeddings(texts=[query])
            query_embedding = embeddings[0]

            # Use LightRAG's entities VDB for semantic search
            results = await self.lightrag.entities_vdb.query(
                query="",  # Not used for vector search
                top_k=limit,
                query_embedding=query_embedding
            )

            # Convert to expected format
            formatted_results = []
            for result in results:
                if result.get("score", 0.0) >= threshold:
                    formatted_results.append({
                        "id": result.get("id", ""),
                        "score": result.get("score", 0.0),
                        "payload": result.get("metadata", {})
                    })

            # Apply entity type filter if specified
            if entity_type:
                formatted_results = [
                    r for r in formatted_results
                    if r.get("payload", {}).get("entity_type") == entity_type
                ]

            return formatted_results

        except Exception as e:
            logger.error(f"Semantic similarity search failed: {str(e)}")
            # Fallback to legacy search
            embeddings = await self.llm.get_embeddings(texts=[query])
            return await self.storage.search_similar_entities(
                embeddings[0], entity_type, limit
            )

    async def hybrid_search(
        self,
        query: str,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Perform hybrid search combining vector and graph search"""
        if not self.lightrag:
            return {"error": "LightRAG not available"}

        try:
            # Get embeddings for vector search
            embeddings = await self.llm.get_embeddings(texts=[query])
            query_embedding = embeddings[0]

            # Perform vector search
            vector_results = await self.lightrag.chunks_vdb.query(
                query="", top_k=limit, query_embedding=query_embedding
            )

            # Extract entities from query for graph search
            query_entities = await self._extract_entities_from_query(query)

            # Perform graph-based search if entities found
            graph_results = []
            if query_entities:
                for entity in query_entities[:3]:  # Limit to first 3 entities
                    entity_context = await self.storage.get_entity_context(entity)
                    if entity_context:
                        graph_results.append({
                            "entity": entity,
                            "context": entity_context
                        })

            # Combine results with weights
            combined_results = {
                "vector_results": vector_results,
                "graph_results": graph_results,
                "vector_weight": vector_weight,
                "graph_weight": graph_weight,
                "query_entities": query_entities
            }

            return combined_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return {"error": str(e)}

    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query text using LLM"""
        try:
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

    async def get_entity_relationships(
        self,
        entity_id: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get entity relationships using LightRAG's graph traversal"""
        if not self.lightrag:
            return {"error": "LightRAG not available"}

        try:
            # This would use LightRAG's graph storage to find relationships
            # For now, return a placeholder
            return {
                "entity_id": entity_id,
                "max_depth": max_depth,
                "relationship_types": relationship_types,
                "relationships": [],
                "related_entities": []
            }

        except Exception as e:
            logger.error(f"Entity relationship query failed: {str(e)}")
            return {"error": str(e)}
