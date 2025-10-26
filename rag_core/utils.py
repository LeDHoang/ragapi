# rag_core/utils.py
import hashlib
from pathlib import Path
from typing import Dict, List, Any

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file for duplicate detection"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (IOError, OSError):
        return ""

def file_mtime(p: Path) -> float:
    return p.stat().st_mtime

def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def cache_key(path: Path, parser: str, method: str, **kwargs) -> str:
    payload = {
        "path": str(path.resolve()),
        "mtime": file_mtime(path),
        "parser": parser,
        "parse_method": method,
        "opts": {k: v for k, v in kwargs.items() if v is not None}
    }
    return md5_hex(str(payload))

def cache_key_with_overlay(path: Path, parser: str, method: str, **kwargs) -> str:
    """Enhanced cache key that includes overlay settings"""
    payload = {
        "path": str(path.resolve()),
        "mtime": file_mtime(path),
        "parser": parser,
        "parse_method": method,
        "opts": {k: v for k, v in kwargs.items() if v is not None},
        "overlay": kwargs.get("export_overlay", False)
    }
    return md5_hex(str(payload))

def content_doc_id(content_list: List[Dict[str, Any]]) -> str:
    sig_parts = []
    for item in content_list:
        t = item.get("type")
        if t == "text" and item.get("text"):
            sig_parts.append(str(item["text"][:5000]))
        elif t == "image" and item.get("img_path"):
            sig_parts.append(f"image:{item['img_path']}")
        elif t == "table" and item.get("table_body"):
            sig_parts.append(f"table:{str(item['table_body'])[:5000]}")
        elif t == "equation" and item.get("text"):
            sig_parts.append(f"equation:{str(item['text'])}")
        else:
            sig_parts.append(str(item))
    return "doc-" + md5_hex("\n".join(sig_parts))

def _coerce_str_list(v) -> List[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [x if isinstance(x, str) else ("" if x is None else str(x)) for x in v]
    # single string or other type
    return [v] if isinstance(v, str) else [str(v)]

def _safe_join_str(v) -> str:
    return " | ".join(_coerce_str_list(v))

# --- multimodal chunk templating (clean output for each type)
def template_chunk(item: Dict[str, Any], description: str) -> str:
    t = item.get("type")
    page_info = f"page={item.get('page_idx', 0)}" if item.get('page_idx') is not None else ""

    if t == "image":
        cap = _safe_join_str(item.get("image_caption"))
        foot = _safe_join_str(item.get("image_footnote"))
        bbox_info = ""
        if item.get("bbox"):
            x1, y1, x2, y2 = item["bbox"]
            bbox_info = f"bbox=[{x1},{y1},{x2},{y2}] "

        return (
            f"[IMAGE] {bbox_info}{page_info}\n"
            f"path={item.get('img_path','')}\n"
            f"caption={cap}\n"
            f"footnote={foot}\n"
            f"summary={description or ''}\n"
        )
    if t == "table":
        cap = _safe_join_str(item.get("table_caption"))
        foot = _safe_join_str(item.get("table_footnote"))
        body = item.get("table_body", "")
        body = body if isinstance(body, str) else ("" if body is None else str(body))

        # Truncate if too long
        max_chars = 5000  # Configurable via environment
        if len(body) > max_chars:
            body = body[:max_chars] + "..."

        bbox_info = ""
        if item.get("bbox"):
            x1, y1, x2, y2 = item["bbox"]
            bbox_info = f"bbox=[{x1},{y1},{x2},{y2}] "

        return (
            f"[TABLE] {bbox_info}{page_info}\n"
            f"caption={cap}\n"
            f"body=\n{body}\n"
            f"footnote={foot}\n"
            f"summary={description or ''}\n"
        )
    if t == "equation":
        fmt = item.get("text_format") or "plain"
        expr = item.get("text","")
        expr = expr if isinstance(expr, str) else ("" if expr is None else str(expr))

        bbox_info = ""
        if item.get("bbox"):
            x1, y1, x2, y2 = item["bbox"]
            bbox_info = f"bbox=[{x1},{y1},{x2},{y2}] "

        return f"[EQUATION] {bbox_info}{page_info}\nformat={fmt}\nexpr={expr}\nsummary={description or ''}\n"
    if t == "text":
        text = item.get("text", "")
        text = text if isinstance(text, str) else ("" if text is None else str(text))

        bbox_info = ""
        if item.get("bbox"):
            x1, y1, x2, y2 = item["bbox"]
            bbox_info = f"bbox=[{x1},{y1},{x2},{y2}] "

        return f"[TEXT] {bbox_info}{page_info}\ncontent=\n{text}\nsummary={description or ''}\n"
    # generic fallback
    return f"[{str(t).upper()}] {page_info}\nsummary={description or ''}\n"
