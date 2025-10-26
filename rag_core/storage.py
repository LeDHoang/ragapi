from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import json, os, time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rag_core.config import AppConfig
from rag_core.utils import md5_hex

@dataclass
class KVStore:
    path: Path
    def initialize(self):
        self.path.mkdir(parents=True, exist_ok=True)
    def _p(self, key: str) -> Path:
        return self.path / f"{key}.json"
    async def get(self, key: str) -> Dict[str,Any] | None:
        p = self._p(key)
        if not p.exists(): return None
        return json.loads(p.read_text())
    async def put(self, key: str, value: Dict[str,Any]):
        self._p(key).write_text(json.dumps(value, ensure_ascii=False))

@dataclass
class TextChunkStore:
    path: Path
    def initialize(self): self.path.mkdir(parents=True, exist_ok=True)
    async def upsert(self, chunks: Dict[str,Dict[str,Any]]):
        for cid, payload in chunks.items():
            (self.path / f"{cid}.txt").write_text(payload["text"], encoding="utf-8")

@dataclass
class SimpleVectorStore:
    path: Path
    def initialize(self): self.path.mkdir(parents=True, exist_ok=True)
    def _vec_p(self): return self.path / "vectors.npy"
    def _meta_p(self): return self.path / "meta.jsonl"

    async def upsert(self, chunks: Dict[str,Dict[str,Any]]):
        # naive bag-of-words embedding; replace with real embeddings
        metas = []
        vecs = []
        for cid, c in chunks.items():
            text = c["text"]
            # toy embedding: hash buckets
            dim = 512
            v = np.zeros((dim,), dtype=np.float32)
            for token in text.split():
                v[hash(token) % dim] += 1.0
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
            metas.append({"chunk_id": cid, "doc_id": c["doc_id"], "type": c["type"],
                          "page_idx": c.get("page_idx"), "preview": text[:400]})
        V = np.array(vecs)
        if self._vec_p().exists():
            V_old = np.load(self._vec_p())
            V = np.vstack([V_old, V])
        np.save(self._vec_p(), V)
        with self._meta_p().open("a", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    async def search(self, query: str, k: int=8) -> List[Dict[str,Any]]:
        # toy query embedding
        dim = 512
        q = np.zeros((dim,), dtype=np.float32)
        for token in query.split():
            q[hash(token) % dim] += 1.0
        q = q / (np.linalg.norm(q) + 1e-9)

        if not self._vec_p().exists(): return []
        V = np.load(self._vec_p())
        sims = cosine_similarity([q], V)[0]
        # read metas
        metas = [json.loads(x) for x in self._meta_p().read_text().splitlines()]
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            m = metas[i]
            results.append({
                "chunk_id": m["chunk_id"],
                "score": float(sims[i]),
                "doc_id": m["doc_id"],
                "type": m["type"],
                "page_idx": m.get("page_idx"),
                "text_preview": m["preview"],
            })
        return results

@dataclass
class DocumentRegistry:
    """Track processed documents to prevent duplicates"""
    path: Path

    def initialize(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.path / "document_registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load the document registry from disk"""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r', encoding='utf-8') as f:
                    self._registry = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._registry = {}
        else:
            self._registry = {}
        self._save_registry()

    def _save_registry(self):
        """Save the document registry to disk"""
        try:
            with open(self._registry_file, 'w', encoding='utf-8') as f:
                json.dump(self._registry, f, ensure_ascii=False, indent=2)
        except IOError:
            pass  # Fail silently if we can't save

    def is_duplicate(self, filename: str, file_hash: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """Check if document is a duplicate and return (is_duplicate, existing_doc_id)"""
        # Check by filename first
        for doc_id, doc_info in self._registry.items():
            if doc_info.get("filename") == filename:
                return True, doc_id

        # Check by content hash
        for doc_id, doc_info in self._registry.items():
            if doc_info.get("file_hash") == file_hash:
                return True, doc_id

        return False, None

    def register_document(self, doc_id: str, filename: str, file_hash: str, file_size: int,
                         content_list: List[Dict[str, Any]], by_type: Dict[str, int]):
        """Register a new processed document"""
        self._registry[doc_id] = {
            "filename": filename,
            "file_hash": file_hash,
            "file_size": file_size,
            "content_list": content_list,
            "by_type": by_type,
            "processed_at": int(time.time()),
            "total_blocks": sum(by_type.values())
        }
        self._save_registry()

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed document"""
        return self._registry.get(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        return [
            {
                "doc_id": doc_id,
                "filename": info["filename"],
                "file_size": info["file_size"],
                "total_blocks": info["total_blocks"],
                "by_type": info["by_type"],
                "processed_at": info["processed_at"]
            }
            for doc_id, info in self._registry.items()
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the registry"""
        if doc_id in self._registry:
            del self._registry[doc_id]
            self._save_registry()
            return True
        return False

    def cleanup_old_entries(self, max_age_days: int = 30):
        """Clean up old registry entries"""
        import time
        cutoff_time = int(time.time()) - (max_age_days * 24 * 60 * 60)

        to_remove = []
        for doc_id, info in self._registry.items():
            if info.get("processed_at", 0) < cutoff_time:
                to_remove.append(doc_id)

        for doc_id in to_remove:
            del self._registry[doc_id]

        if to_remove:
            self._save_registry()

@dataclass
class StorageBundle:
    cache: KVStore
    text_chunks: TextChunkStore
    vectors: SimpleVectorStore
    documents: DocumentRegistry

    @staticmethod
    def initialize(cfg: AppConfig) -> "StorageBundle":
        root = Path(cfg.working_dir)
        cache = KVStore(root / "kv" / "parse_cache"); cache.initialize()
        text  = TextChunkStore(root / "text_chunks"); text.initialize()
        vec   = SimpleVectorStore(root / "vectors"); vec.initialize()
        docs  = DocumentRegistry(root / "kv" / "document_registry"); docs.initialize()
        return StorageBundle(cache, text, vec, docs)
