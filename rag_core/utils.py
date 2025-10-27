from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import hashlib
import base64
import time
from pathlib import Path
import numpy as np
from .config import config

logger = logging.getLogger(__name__)

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash ID for content"""
    hash_md5 = hashlib.md5(content.encode()).hexdigest()
    return f"{prefix}{hash_md5}"

def save_numpy_array(array: np.ndarray, file_path: Path):
    """Save numpy array to file"""
    np.save(str(file_path), array)

def load_numpy_array(file_path: Path) -> np.ndarray:
    """Load numpy array from file"""
    return np.load(str(file_path))

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def save_json(data: Any, file_path: Path):
    """Save data as JSON"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(file_path: Path) -> Any:
    """Load data from JSON"""
    with open(file_path) as f:
        return json.load(f)

class VectorIndex:
    def __init__(self, vectors_dir: Path):
        self.vectors_dir = vectors_dir
        self.vectors_file = vectors_dir / "vectors.npy"
        self.meta_file = vectors_dir / "meta.jsonl"
        self.vectors = None
        self.metadata = []
        self._load_index()
    
    def _load_index(self):
        """Load vector index from disk"""
        if self.vectors_file.exists():
            self.vectors = load_numpy_array(self.vectors_file)
        
        if self.meta_file.exists():
            with open(self.meta_file) as f:
                self.metadata = [json.loads(line) for line in f]
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ):
        """Add vectors to index"""
        vectors_array = np.array(vectors)
        
        if self.vectors is None:
            self.vectors = vectors_array
        else:
            self.vectors = np.vstack([self.vectors, vectors_array])
        
        self.metadata.extend(metadata)
        
        # Save to disk
        save_numpy_array(self.vectors, self.vectors_file)
        with open(self.meta_file, "a") as f:
            for meta in metadata:
                f.write(json.dumps(meta) + "\n")
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.vectors is None or len(self.vectors) == 0:
            return []
        
        # Convert query to numpy array
        query = np.array(query_vector)
        
        # Calculate cosine similarity
        similarities = np.dot(self.vectors, query) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query)
        )
        
        # Get top matches
        top_indices = np.argsort(similarities)[-limit:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({
                    "score": score,
                    "metadata": self.metadata[idx]
                })
        
        return results

class ChunkManager:
    def __init__(self, chunks_dir: Path):
        self.chunks_dir = chunks_dir
        self.index_file = chunks_dir / "index.jsonl"
        self.index = {}
        self._load_index()
    
    def _load_index(self):
        """Load chunk index from disk"""
        if self.index_file.exists():
            with open(self.index_file) as f:
                for line in f:
                    chunk = json.loads(line)
                    self.index[chunk["id"]] = chunk
    
    def add_chunk(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any]
    ):
        """Add chunk to storage"""
        chunk_data = {
            "id": chunk_id,
            "content": content,
            **metadata
        }
        
        # Save chunk
        chunk_file = self.chunks_dir / f"{chunk_id}.txt"
        with open(chunk_file, "w") as f:
            f.write(content)
        
        # Update index
        self.index[chunk_id] = chunk_data
        with open(self.index_file, "a") as f:
            f.write(json.dumps(chunk_data) + "\n")
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID"""
        if chunk_id not in self.index:
            return None
        
        chunk_data = self.index[chunk_id]
        chunk_file = self.chunks_dir / f"{chunk_id}.txt"
        
        if chunk_file.exists():
            with open(chunk_file) as f:
                chunk_data["content"] = f.read()
            return chunk_data
        
        return None
    
    def get_chunks_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        return [
            self.get_chunk(chunk_id)
            for chunk_id, chunk in self.index.items()
            if chunk.get("doc_id") == doc_id
        ]

class DocumentRegistry:
    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_file = registry_dir / "document_registry.json"
        self.registry = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load document registry from disk"""
        if self.registry_file.exists():
            self.registry = load_json(self.registry_file)
    
    def _save_registry(self):
        """Save document registry to disk"""
        save_json(self.registry, self.registry_file)
    
    def register_document(
        self,
        doc_id: str,
        file_path: str,
        metadata: Dict[str, Any]
    ):
        """Register a document"""
        self.registry[doc_id] = {
            "file_path": file_path,
            "registered_at": time.time(),
            **metadata
        }
        self._save_registry()
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        return self.registry.get(doc_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all registered documents"""
        return [
            {"doc_id": doc_id, **metadata}
            for doc_id, metadata in self.registry.items()
        ]
    
    def remove_document(self, doc_id: str):
        """Remove document from registry"""
        if doc_id in self.registry:
            del self.registry[doc_id]
            self._save_registry()