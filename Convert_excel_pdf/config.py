"""
Configuration helpers for the Collabora-based Excel→PDF toolkit.

Expose knobs for sheet splitting and debug artifact generation so the main
API can keep copies of intermediate files during conversion.
"""

from __future__ import annotations

import os
from pathlib import Path


def _to_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Root directory for split artifacts (defaults to the local split/ folder)
_default_split_root = Path(__file__).resolve().parent / "split"
SPLIT_OUTPUT_ROOT = Path(os.getenv("CONVERT_SPLIT_OUTPUT_ROOT", str(_default_split_root))).resolve()

# Maximum number of sheets per split workbook
MAX_SHEETS_PER_SPLIT = int(os.getenv("CONVERT_SPLIT_MAX_SHEETS", "10"))

# Whether to keep debug artifacts (split Excel files + PDFs)
GENERATE_DEBUG_ARTIFACTS = _to_bool(os.getenv("CONVERT_GENERATE_DEBUG_ARTIFACTS", "true"), True)

