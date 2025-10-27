from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import logging
import json
import time
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from neo4j import Driver

from .config import config
from .schemas import EntityNode, EntityRelation

logger = logging.getLogger(__name__)

# Optional imports - handle gracefully if not available
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    logger.warning("Neo4j not available. Graph database features will be disabled.")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None
    logger.warning("Qdrant not available. Vector database features will be disabled.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    logger.warning("Redis not available. Cache features will be disabled.")

class Neo4jGraph:
    def __init__(self):
        self.config = config
        self.driver = self._init_driver()
        if self.driver:
            self._ensure_constraints()

    def _init_driver(self) -> Optional[Any]:
        """Initialize Neo4j driver with error handling"""
        if not NEO4J_AVAILABLE:
            logger.info("Neo4j not available, graph operations disabled")
            return None

        try:
            uri = self.config.GRAPH_DB
            if "://" not in uri:
                uri = f"neo4j://{uri}"

            driver = GraphDatabase.driver(
                uri,
                auth=(self.config.GRAPH_DB_USER, self.config.GRAPH_DB_PASSWORD)
            )
            driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.config.GRAPH_DB}")
            return driver
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j at {self.config.GRAPH_DB}: {e}")
            logger.warning("Graph database will be disabled.")
            return None

    def _ensure_constraints(self):
        """Ensure necessary constraints exist in the graph database"""
        if not self.driver:
            return

        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Failed to create constraint: {e}")

    async def create_document(self, doc_id: str, file_path: str, metadata: Dict[str, Any]):
        """Create document node in graph"""
        if not self.driver:
            logger.debug("Graph database not available, skipping document creation")
            return

        query = """
        MERGE (d:Document {
            doc_id: $doc_id
        })
        SET d.file_path = $file_path,
            d.file_type = $file_type,
            d.total_pages = $total_pages,
            d.chunks_count = $chunks_count,
            d.entities_count = $entities_count,
            d.created_at = $created_at
        """

        params = {
            "doc_id": doc_id,
            "file_path": file_path,
            "file_type": metadata.get("file_type", ""),
            "total_pages": metadata.get("total_pages", 0),
            "chunks_count": metadata.get("chunks_count", 0),
            "entities_count": metadata.get("entities_count", 0),
            "created_at": metadata.get("processed_at", time.time())
        }

        with self.driver.session() as session:
            session.run(query, params)

    async def create_chunk(self, chunk_id: str, content: str, doc_id: str, chunk_type: str = "text", page_idx: Optional[int] = None):
        """Create chunk node in graph"""
        if not self.driver:
            logger.debug("Graph database not available, skipping chunk creation")
            return

        query = """
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c:Chunk {
            chunk_id: $chunk_id
        })
        SET c.content = $content,
            c.chunk_type = $chunk_type,
            c.page_idx = $page_idx,
            c.created_at = $created_at
        MERGE (d)-[:CONTAINS]->(c)
        """

        params = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "content": content,
            "chunk_type": chunk_type,
            "page_idx": page_idx,
            "created_at": time.time()
        }

        with self.driver.session() as session:
            session.run(query, params)

    async def create_entity(self, entity: EntityNode):
        """Create entity node in graph following RAG-Anything approach"""
        if not self.driver:
            logger.debug("Graph database not available, skipping entity creation")
            return

        # Create entity node
        entity_query = """
        MERGE (e:Entity {
            entity_id: $entity_id
        })
        SET e.entity_type = $entity_type,
            e.name = $name,
            e.description = $description,
            e.source_id = $source_id,
            e.file_path = $file_path,
            e.created_at = $created_at
        """

        # Link to source chunk
        chunk_query = """
        MATCH (e:Entity {entity_id: $entity_id})
        MATCH (c:Chunk {chunk_id: $source_id})
        MERGE (c)-[:MENTIONS]->(e)
        """

        with self.driver.session() as session:
            session.run(entity_query, entity.dict())
            session.run(chunk_query, {"entity_id": entity.entity_id, "source_id": entity.source_id})
    
    async def create_relation(self, relation: EntityRelation):
        """Create relationship between entities following RAG-Anything approach"""
        if not self.driver:
            logger.debug("Graph database not available, skipping relation creation")
            return

        # Create relationship based on type
        relation_type = relation.relation_type.upper()

        if relation_type == "BELONGS_TO":
            query = """
            MATCH (src:Entity {entity_id: $src_id})
            MATCH (tgt:Entity {entity_id: $tgt_id})
            MERGE (src)-[r:BELONGS_TO {
                description: $description,
                keywords: $keywords,
                source_id: $source_id,
                weight: $weight,
                file_path: $file_path,
                created_at: $created_at
            }]->(tgt)
            """
        elif relation_type == "REFERENCES":
            query = """
            MATCH (src:Entity {entity_id: $src_id})
            MATCH (tgt:Entity {entity_id: $tgt_id})
            MERGE (src)-[r:REFERENCES {
                description: $description,
                keywords: $keywords,
                source_id: $source_id,
                weight: $weight,
                file_path: $file_path,
                created_at: $created_at
            }]->(tgt)
            """
        elif relation_type == "APPEARS_ON":
            query = """
            MATCH (src:Entity {entity_id: $src_id})
            MATCH (tgt:Chunk {chunk_id: $tgt_id})
            MERGE (src)-[r:APPEARS_ON {
                description: $description,
                keywords: $keywords,
                source_id: $source_id,
                weight: $weight,
                file_path: $file_path,
                created_at: $created_at
            }]->(tgt)
            """
        else:
            # Generic relationship
            query = """
            MATCH (src:Entity {entity_id: $src_id})
            MATCH (tgt:Entity {entity_id: $tgt_id})
            MERGE (src)-[r:RELATED_TO {
                relation_type: $relation_type,
                description: $description,
                keywords: $keywords,
                source_id: $source_id,
                weight: $weight,
                file_path: $file_path,
                created_at: $created_at
            }]->(tgt)
            """

        params = relation.dict()
        params["created_at"] = time.time()

        with self.driver.session() as session:
            session.run(query, params)
    
    async def get_entity_neighbors(
        self,
        entity_id: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Get entity neighbors up to max_depth"""
        if not self.driver:
            logger.debug("Graph database not available, returning empty neighbors")
            return []
        
        relation_filter = ""
        if relation_types:
            types_list = [f"'{t}'" for t in relation_types]
            relation_filter = f"WHERE r.relation_type IN [{', '.join(types_list)}]"
        
        query = f"""
        MATCH path = (src:Entity {{entity_id: $entity_id}})-[r*1..{max_depth}]->(n:Entity)
        {relation_filter}
        RETURN path
        """
        
        with self.driver.session() as session:
            result = session.run(query, {"entity_id": entity_id})
            
            neighbors = []
            for record in result:
                path = record["path"]
                path_data = []
                
                for node in path.nodes:
                    path_data.append({
                        "entity_id": node["entity_id"],
                        "entity_type": node["entity_type"],
                        "name": node["name"],
                        "description": node["description"]
                    })
                
                neighbors.append({
                    "path": path_data,
                    "length": len(path_data) - 1
                })
            
            return neighbors
    
    async def get_shortest_path(
        self,
        src_id: str,
        tgt_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities"""
        if not self.driver:
            logger.debug("Graph database not available, returning None for shortest path")
            return None
        
        query = """
        MATCH path = shortestPath((src:Entity {entity_id: $src_id})-[*]->(tgt:Entity {entity_id: $tgt_id}))
        RETURN path
        """
        
        with self.driver.session() as session:
            result = session.run(query, {"src_id": src_id, "tgt_id": tgt_id})
            
            for record in result:
                path = record["path"]
                path_data = []
                
                for node in path.nodes:
                    path_data.append({
                        "entity_id": node["entity_id"],
                        "entity_type": node["entity_type"],
                        "name": node["name"],
                        "description": node["description"]
                    })
                
                return path_data
            
            return None

class QdrantVectorStore:
    def __init__(self):
        self.config = config
        self.client = None
        self._init_client()
        if self.client:
            self._ensure_collections()
    
    def _init_client(self) -> Optional[Any]:
        """Initialize Qdrant client with error handling"""
        # Skip Qdrant initialization if using local storage
        if self.config.VECTOR_DB.startswith("local://"):
            logger.info("Using local vector storage instead of Qdrant")
            self.client = None
            return None
            
        try:
            url = self.config.VECTOR_DB.replace("qdrant://", "")
            host, port = url.split(":")
            self.client = QdrantClient(host=host, port=int(port))
            # Test connection
            self.client.get_collections()
            return self.client
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant at {self.config.VECTOR_DB}: {e}")
            logger.warning("Vector store will be disabled. Please start Qdrant server.")
            self.client = None
            return None
    
    def _ensure_collections(self):
        """Ensure required collections exist"""
        collections = {
            "chunks": {
                "size": self.config.EMBEDDING_DIM,
                "distance": Distance.COSINE
            },
            "entities": {
                "size": self.config.EMBEDDING_DIM,
                "distance": Distance.COSINE
            }
        }
        
        for name, params in collections.items():
            try:
                self.client.get_collection(name)
            except:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=params["size"],
                        distance=params["distance"]
                    )
                )
    
    async def store_vectors(
        self,
        collection: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """Store vectors with metadata"""
        if not self.client:
            logger.debug("Qdrant not available, skipping vector storage")
            return
        
        if not ids:
            ids = [str(i) for i in range(len(vectors))]
        
        points = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=vector,
                    payload=meta
                )
            )
        
        self.client.upsert(
            collection_name=collection,
            points=points
        )
    
    async def search_vectors(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if not self.client:
            logger.debug("Qdrant not available, returning empty search results")
            return []
        
        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]

class RedisCache:
    def __init__(self):
        self.config = config
        self.client = self._init_client()

    def _init_client(self) -> Optional[Any]:
        """Initialize Redis client with error handling"""
        if not REDIS_AVAILABLE:
            logger.info("Redis not available, cache operations disabled")
            return None

        try:
            url = self.config.CACHE_DB
            return redis.from_url(url)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {self.config.CACHE_DB}: {e}")
            logger.warning("Cache operations will be disabled.")
            return None
    
    async def set_cache(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ):
        """Set cache value"""
        if not self.client:
            logger.debug("Redis not available, skipping cache set")
            return

        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        self.client.set(key, value, ex=expire)
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.client:
            logger.debug("Redis not available, returning None from cache")
            return None

        value = self.client.get(key)
        if not value:
            return None

        try:
            return json.loads(value)
        except:
            return value.decode()

    async def delete_cache(self, key: str):
        """Delete cache value"""
        if not self.client:
            logger.debug("Redis not available, skipping cache delete")
            return

        self.client.delete(key)

class StorageManager:
    def __init__(self):
        self.graph = Neo4jGraph()
        self.vectors = QdrantVectorStore()
        self.cache = RedisCache()

    async def store_document(
        self,
        doc_id: str,
        file_path: str,
        metadata: Dict[str, Any]
    ):
        """Store document in graph"""
        await self.graph.create_document(doc_id, file_path, metadata)

    async def store_chunk(
        self,
        chunk_id: str,
        content: str,
        doc_id: str,
        chunk_type: str = "text",
        page_idx: Optional[int] = None,
        vector: Optional[List[float]] = None
    ):
        """Store chunk in graph and vector store"""

        # Store in graph
        await self.graph.create_chunk(chunk_id, content, doc_id, chunk_type, page_idx)

        # Store vector if provided
        if vector is not None:
            await self.vectors.store_vectors(
                collection="chunks",
                vectors=[vector],
                metadata=[{
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": content,
                    "chunk_type": chunk_type,
                    "page_idx": page_idx
                }],
                ids=[chunk_id]
            )

    async def store_entity(
        self,
        entity: EntityNode,
        vector: Optional[List[float]] = None
    ):
        """Store entity in graph and vector store following RAG-Anything approach"""

        # Store in graph
        await self.graph.create_entity(entity)

        # Store vector if provided
        if vector is not None:
            await self.vectors.store_vectors(
                collection="entities",
                vectors=[vector],
                metadata=[entity.dict()],
                ids=[entity.entity_id]
            )
    
    async def store_relation(
        self,
        relation: EntityRelation,
        vector: Optional[List[float]] = None
    ):
        """Store relation in graph and vector store"""
        
        # Store in graph
        await self.graph.create_relation(relation)
        
        # Store vector if provided
        if vector is not None:
            await self.vectors.store_vectors(
                collection="relations",
                vectors=[vector],
                metadata=[relation.dict()],
                ids=[f"{relation.src_id}_{relation.tgt_id}"]
            )
    
    async def search_similar_entities(
        self,
        query_vector: List[float],
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar entities"""
        
        filter = None
        if entity_type:
            filter = {"entity_type": entity_type}
        
        return await self.vectors.search_vectors(
            collection="entities",
            query_vector=query_vector,
            limit=limit,
            filter=filter
        )
    
    async def get_entity_context(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get entity context from graph"""
        
        # Get entity neighbors
        neighbors = await self.graph.get_entity_neighbors(
            entity_id=entity_id,
            max_depth=max_depth
        )
        
        # Group by type
        context = {
            "neighbors": neighbors,
            "by_type": {}
        }
        
        for path in neighbors:
            for node in path["path"]:
                node_type = node["entity_type"]
                if node_type not in context["by_type"]:
                    context["by_type"][node_type] = []
                if node not in context["by_type"][node_type]:
                    context["by_type"][node_type].append(node)
        
        return context

    async def extract_and_store_entities(
        self,
        content: str,
        chunk_id: str,
        doc_id: str,
        file_path: str,
        content_type: str = "text"
    ):
        """Extract entities from content and store them in the graph"""
        if not self.graph.driver:
            logger.debug("Graph database not available, skipping entity extraction")
            return []

        # Use LLM to extract entities following RAG-Anything approach
        from .llm_unified import UnifiedLLM

        try:
            llm = UnifiedLLM()

            # Check if LLM is properly configured
            if not hasattr(llm, 'openai_client') or llm.openai_client is None:
                logger.info("LLM not configured, using fallback entity extraction")
                return await self._fallback_entity_extraction(content, chunk_id, file_path)

            prompt = f"""
            Extract key entities from the following content. Focus on:

            1. **Named entities**: People, organizations, locations, products, concepts
            2. **Technical terms**: Domain-specific terminology, acronyms, proper nouns
            3. **Key concepts**: Important ideas, processes, or topics mentioned
            4. **Visual elements**: For multimodal content, identify visual components

            Content type: {content_type}
            Content:
            {content}

            Return a JSON array of entities with the following structure:
            [
                {{
                    "name": "Entity Name",
                    "type": "person|organization|location|product|concept|technical_term|visual_element",
                    "description": "Brief description of what this entity represents",
                    "confidence": 0.8
                }}
            ]

            Only extract entities that are clearly identifiable and relevant.
            """

            response = await llm.generate_text(
                prompt=prompt,
                system_prompt="You are an expert at extracting structured information from documents."
            )

            # Parse the response to extract entities
            import json
            import re

            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group())
            else:
                # Fallback: try to parse the whole response
                entities_data = json.loads(response)

            # Create and store entities
            stored_entities = []
            for entity_data in entities_data:
                entity = EntityNode(
                    entity_id=f"{entity_data['name'].replace(' ', '_').lower()}_{chunk_id}",
                    entity_type=entity_data['type'],
                    name=entity_data['name'],
                    description=entity_data['description'],
                    source_id=chunk_id,
                    file_path=file_path,
                    created_at=time.time()
                )

                # Create embedding for entity
                entity_text = f"{entity.name}\n{entity.description}"
                embeddings = await llm.get_embeddings(texts=[entity_text])
                entity_vector = embeddings[0] if embeddings else None

                await self.store_entity(entity, entity_vector)
                stored_entities.append(entity)

                # Create relationship between chunk and entity
                relation = EntityRelation(
                    src_id=chunk_id,
                    tgt_id=entity.entity_id,
                    relation_type="APPEARS_ON",
                    description=f"Entity {entity.name} appears in chunk",
                    keywords=f"{entity.name},{entity.entity_type}",
                    source_id=chunk_id,
                    weight=entity_data.get('confidence', 0.8),
                    file_path=file_path
                )
                await self.store_relation(relation)

            logger.info(f"Extracted and stored {len(stored_entities)} entities from chunk {chunk_id}")
            return stored_entities

        except Exception as e:
            logger.warning(f"Failed to extract entities from chunk {chunk_id}: {e}")
            return await self._fallback_entity_extraction(content, chunk_id, file_path)

    async def _fallback_entity_extraction(self, content: str, chunk_id: str, file_path: str):
        """Fallback entity extraction using simple keyword matching"""
        import re

        # Simple keyword-based entity extraction
        entities_data = []

        # Common entity patterns
        patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'person'),  # Proper names like "John Smith"
            (r'\b[A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd|Company|Corporation)\b', 'organization'),  # Company names
            (r'\b[A-Z][a-zA-Z]+ (University|College|Institute|School)\b', 'organization'),  # Educational institutions
            (r'\b[A-Z][a-zA-Z]+ (AI|ML|API|SDK|DB|OS)\b', 'technical_term'),  # Technical acronyms
            (r'\b(Machine Learning|Artificial Intelligence|Deep Learning|Neural Network|Computer Vision)\b', 'concept'),  # Common AI concepts
        ]

        for pattern, entity_type in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 2:  # Filter out very short matches
                    entities_data.append({
                        "name": match,
                        "type": entity_type,
                        "description": f"Extracted {entity_type} from text",
                        "confidence": 0.6
                    })

        # Create and store entities
        stored_entities = []
        for entity_data in entities_data:
            entity = EntityNode(
                entity_id=f"{entity_data['name'].replace(' ', '_').lower()}_{chunk_id}",
                entity_type=entity_data['type'],
                name=entity_data['name'],
                description=entity_data['description'],
                source_id=chunk_id,
                file_path=file_path,
                created_at=time.time()
            )

            await self.store_entity(entity)
            stored_entities.append(entity)

            # Create relationship between chunk and entity
            relation = EntityRelation(
                src_id=chunk_id,
                tgt_id=entity.entity_id,
                relation_type="APPEARS_ON",
                description=f"Entity {entity.name} appears in chunk",
                keywords=f"{entity.name},{entity.entity_type}",
                source_id=chunk_id,
                weight=entity_data.get('confidence', 0.6),
                file_path=file_path
            )
            await self.store_relation(relation)

        logger.info(f"Fallback extracted and stored {len(stored_entities)} entities from chunk {chunk_id}")
        return stored_entities

    async def find_entity_relationships(self, doc_id: str):
        """Find and create relationships between entities in a document"""
        if not self.graph.driver:
            logger.debug("Graph database not available, skipping relationship extraction")
            return

        # Get all entities for the document
        query = """
        MATCH (e:Entity)-[r1]-(c:Chunk)-[r2]-(d:Document {doc_id: $doc_id})
        RETURN e.entity_id, e.entity_type, e.name, e.description, c.chunk_id, c.page_idx
        """

        with self.graph.driver.session() as session:
            result = session.run(query, {"doc_id": doc_id})
            entities = []
            for record in result:
                entities.append({
                    "entity_id": record["e.entity_id"],
                    "entity_type": record["e.entity_type"],
                    "name": record["e.name"],
                    "description": record["e.description"],
                    "chunk_id": record["c.chunk_id"],
                    "page_idx": record["c.page_idx"]
                })

        # Create relationships between related entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities are related (same page, similar types, etc.)
                if self._entities_are_related(entity1, entity2):
                    relation = EntityRelation(
                        src_id=entity1["entity_id"],
                        tgt_id=entity2["entity_id"],
                        relation_type="REFERENCES",
                        description=f"Entities {entity1['name']} and {entity2['name']} are related",
                        keywords=f"{entity1['name']},{entity2['name']},{entity1['entity_type']},{entity2['entity_type']}",
                        source_id=entity1["chunk_id"],
                        weight=0.7,
                        file_path=doc_id
                    )
                    await self.store_relation(relation)

    def _entities_are_related(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entities are related based on RAG-Anything logic"""
        # Same page
        if entity1["page_idx"] == entity2["page_idx"]:
            return True

        # Similar entity types
        if entity1["entity_type"] == entity2["entity_type"]:
            return True

        # Check semantic similarity in names
        name1_words = set(entity1["name"].lower().split())
        name2_words = set(entity2["name"].lower().split())
        overlap = len(name1_words.intersection(name2_words))
        if overlap > 0 and (overlap / min(len(name1_words), len(name2_words))) > 0.3:
            return True

        return False