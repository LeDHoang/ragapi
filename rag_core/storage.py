from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json, os
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
class StorageBundle:
    cache: KVStore
    text_chunks: TextChunkStore
    vectors: SimpleVectorStore

    @staticmethod
    def initialize(cfg: AppConfig) -> "StorageBundle":
        root = Path(cfg.working_dir)
        cache = KVStore(root / "kv" / "parse_cache"); cache.initialize()
        text  = TextChunkStore(root / "text_chunks"); text.initialize()
        vec   = SimpleVectorStore(root / "vectors"); vec.initialize()
        return StorageBundle(cache, text, vec)
